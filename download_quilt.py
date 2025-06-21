import os
from huggingface_hub import snapshot_download

def download_model(repo_id, save_dir):
    """
    Downloads a Hugging Face model and saves it to a local directory.
    """
    print(f"📥 Downloading model from Hugging Face ({repo_id})")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=os.path.join(save_dir, ".hf_cache"),
        local_dir=save_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5"]
    )
    print(f"Model saved to: {save_dir}")

if __name__ == "__main__":
    # === Paths ===
    base_save_path = "/project/hnguyen2/mvu9/checkpoints"

    # === QUILT ===
    quilt_repo = "TriDao/quilt-1b"
    quilt_dir = os.path.join(base_save_path, "quilt_checkpoints")
    os.makedirs(quilt_dir, exist_ok=True)
    download_model(quilt_repo, quilt_dir)
