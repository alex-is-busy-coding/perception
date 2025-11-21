import logging
import opensmile
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    A class to extract acoustic and conversational features from audio segments.
    """
    def __init__(self):
        """
        Initializes the FeatureExtractor with the eGeMAPS feature set.
        """
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

    def extract_opensmile_features(self, audio_path: str) -> pd.DataFrame:
        """
        Extracts eGeMAPS features for a specific segment of an audio file.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            pd.DataFrame: A DataFrame containing the 88 eGeMAPS features for the segment.
        """
        try:
            features = self.smile.process_file(audio_path)
            return features
        except Exception as e:
            logger.exception(f"Could not extract openSMILE features audio from '{audio_path}': {e}")
            return pd.DataFrame()
