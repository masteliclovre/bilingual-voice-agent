import os
from huggingface_hub import snapshot_download

# The full, original model repository. This contains the necessary config files.
MODEL_ID = "GoranS/whisper-large-v3-turbo-hr-parla"
# The destination directory inside the Docker container.
DESTINATION_DIR = "./models/"

print(f"--- Starting model download ---")
print(f"Model: {MODEL_ID}")
print(f"Destination: {DESTINATION_DIR}")

# Download the model files. This happens during the 'docker build' step.
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=DESTINATION_DIR,
    local_dir_use_symlinks=False
)

print(f"--- Model download complete ---")


