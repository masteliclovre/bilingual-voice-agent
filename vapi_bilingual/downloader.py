from huggingface_hub import snapshot_download
# replace with the exact repo id you’re using:
repo_id = "Gorans/whisper-large-v3-turbo-hr-parla-ctranslate2"
local_dir = snapshot_download(repo_id=repo_id, local_dir="whisper-large-v3-turbo-hr-parla-ctranslate2", local_dir_use_symlinks=False)
print("Downloaded to:", local_dir)
