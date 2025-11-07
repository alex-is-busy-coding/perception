process:
	@echo "Processing raw audio data..."
	@if [ -z "$${HUGGING_FACE_HUB_TOKEN}" ]; then \
		echo "Error: HUGGING_FACE_HUB_TOKEN is not set."; \
		echo "Please set it (e.g., export HUGGING_FACE_HUB_TOKEN='your_token')"; \
		exit 1; \
	fi
	uv run main.py

train:
	@echo "Training model..."
	uv run "model_training/train.py"

clean:
	@echo "Cleaning up..."
	rm -rf lightning_logs/

tensorboard:
	@echo "Starting TensorBoard..."
	uv run tensorboard --logdir=lightning_logs/