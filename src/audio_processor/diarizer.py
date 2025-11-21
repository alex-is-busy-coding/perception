import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

logger = logging.getLogger(__name__)

class SpeakerDiarizer:
    """
    Processes audio files using WhisperX to perform automatic speech recognition (ASR),
    word-level alignment, and speaker diarization.
    """

    def __init__(
        self,
        hf_token: str = None,
        device: str = None,
        asr_model_name: str = "large-v3",
        diarize_model_name: str = "pyannote/speaker-diarization-3.1",
        batch_size: int = 16
    ):
        """
        Initializes the SpeakerDiarizer with ASR and Diarization models.

        Args:
            hf_token (str, optional): Hugging Face authentication token required for 
                Pyannote. If None, attempts to read from 'HF_TOKEN' environment variable.
            device (str, optional): Computation device to use ('cuda' or 'cpu'). 
                If None, automatically selects 'cuda' if available.
            asr_model_name (str, optional): Name of the Whisper model to load. 
                Defaults to "large-v3".
            diarize_model_name (str, optional): Name of the Pyannote model to load. 
                Defaults to "pyannote/speaker-diarization-3.1".
            batch_size (int, optional): Batch size for the ASR transcription process. 
                Defaults to 16.

        Raises:
            ValueError: If the Hugging Face token is missing.
        """
        if hf_token is None:
            hf_token = os.getenv("HF_TOKEN")
            if hf_token is None:
                raise ValueError(
                    "Hugging Face token not found. "
                    "Pass it to the constructor or set the HF_TOKEN environment variable."
                )

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.batch_size = batch_size
        
        logger.info(f"Using device: {self.device} with compute type: {self.compute_type}")
        logger.info(f"Loading ASR model: {asr_model_name}...")
        
        self.model = whisperx.load_model(
            asr_model_name,
            self.device,
            compute_type=self.compute_type
        )

        logger.info(f"Loading Diarization model: {diarize_model_name}...")
        
        self.diarize_model = DiarizationPipeline(
            model_name=diarize_model_name,
            use_auth_token=hf_token,
            device=self.device
        )

        self.model_a = None
        self.metadata = None
        
        logger.info("Diarizer initialized successfully.")

    def _load_alignment_model(self, language_code: str) -> None:
        """
        Loads and caches the alignment model for the specified language.

        If the model for the requested language is already loaded, this method 
        performs no action.

        Args:
            language_code (str): The ISO language code (e.g., 'en', 'fr') detected 
                during transcription.
        """
        if self.model_a is not None and self.metadata["language"] == language_code:
            return

        logger.info(f"Loading alignment model for language: {language_code}...")
        
        self.model_a, self.metadata = whisperx.load_align_model(
            language_code=language_code, device=self.device
        )
        self.metadata["language"] = language_code

    def _create_dataframe(self, final_result: dict) -> pd.DataFrame:
        """
        Transforms the raw WhisperX result dictionary into a pandas DataFrame.

        Args:
            final_result (dict): The dictionary returned by whisperx.assign_word_speakers containing
                segments and word-level details.

        Returns:
            pd.DataFrame: A DataFrame containing columns ['start', 'end', 'word', 'speaker', 'score'].
                If diarization fails, the 'speaker' column defaults to 'UNKNOWN'. 
                Returns an empty DataFrame if no words are found.
        """
        all_words = []
        for segment in final_result["segments"]:
            if "words" in segment:
                all_words.extend(segment["words"])

        if not all_words:
            logger.warning("No words found in the result.")
            return pd.DataFrame()

        df = pd.DataFrame(all_words)

        if 'speaker' in df.columns:
            df = df[['start', 'end', 'word', 'speaker', 'score']]
            df['speaker'] = df['speaker'].fillna('UNKNOWN')
        else:
            logger.warning("'speaker' column not found. Diarization may have failed.")
            df = df[['start', 'end', 'word', 'score']]
            df['speaker'] = 'UNKNOWN'

        return df

    def _generate_plot(self, df: pd.DataFrame, output_path: str):
        """
        Generates a Gantt-style visualization of the speaker diarization.
        """
        if df.empty or 'speaker' not in df.columns:
            logger.warning("Cannot generate plot: DataFrame is empty or missing speaker info.")
            return

        plt.figure(figsize=(20, 4))
        
        speakers = sorted(df['speaker'].unique())
        cmap = plt.get_cmap("tab20")
        speaker_colors = {spk: cmap(i % 20) for i, spk in enumerate(speakers)}
        
        min_time = df['start'].min()
        max_time = df['end'].max()

        for spk in speakers:
            spk_data = df[df['speaker'] == spk]
            durations = spk_data['end'] - spk_data['start']
            plt.barh(
                y=[spk] * len(spk_data), 
                width=durations, 
                left=spk_data['start'], 
                color=speaker_colors[spk], 
                edgecolor='none',
                height=0.4
            )

        plt.xlabel("Time (seconds)")
        plt.ylabel("Speakers")
        plt.title("Speaker Diarization Timeline")
        plt.xlim(min_time, max_time)
        plt.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Diarization plot saved to: {output_path}")

    def process(self, audio_file_path: str, output_path: str = None, plot_output_path: str = None) -> pd.DataFrame:
        """
        Executes the full pipeline and optionally saves CSV and Plot.

        Args:
            audio_file_path (str): Input audio file.
            output_path (str, optional): Path to save CSV.
            plot_output_path (str, optional): Path to save the visualization PNG.
        """
        logger.info(f"Loading audio from: {audio_file_path}...")
        audio = whisperx.load_audio(audio_file_path)

        logger.info("Transcribing audio...")
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        language_code = result["language"]

        self._load_alignment_model(language_code)

        logger.info("Aligning transcription...")
        aligned_result = whisperx.align(
            result["segments"],
            self.model_a,
            self.metadata,
            audio,
            self.device,
            return_char_alignments=False
        )

        logger.info("Diarizing speakers...")
        diarize_segments = self.diarize_model(audio)

        logger.info("Assigning speakers to words...")
        final_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)

        logger.info("Creating DataFrame...")
        df = self._create_dataframe(final_result)
        
        if output_path:
            logger.info(f"Saving results to CSV at: {output_path}")
            df.to_csv(output_path, index=False)

        if plot_output_path:
            logger.info("Generating diarization plot...")
            self._generate_plot(df, plot_output_path)
        
        logger.info("Processing Complete.")
        return df