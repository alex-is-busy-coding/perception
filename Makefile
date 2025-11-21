RUN = uv run
ENV = PYTHONPATH=.

.PHONY: process-audio process-video process train visualize clean tensorboard

process-audio:
	@echo "Processing audio data..."
	$(ENV) $(RUN) scripts/process_audio.py

process-video:
	@echo "Processing video data..."
	$(ENV) $(RUN) scripts/process_video.py

process: process-audio process-video

train:
	@echo "Training model..."
	$(ENV) $(RUN) scripts/train.py

visualize:
	@echo "Generating embeddings and visualization..."
	$(ENV) $(RUN) scripts/visualize.py

tensorboard:
	@echo "Starting TensorBoard..."
	$(ENV) $(RUN) tensorboard --logdir=tensorboard_logs/

clean:
	@echo "Cleaning up..."
	rm -rf lightning_logs/
	rm -rf tensorboard_logs/
	find . -type d -name "__pycache__" -exec rm -rf {} +