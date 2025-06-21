import os
from huggingface_hub import snapshot_download

def download_conch_model(save_dir):
    """
    Downloads the CONCH model checkpoint from Hugging Face
    and saves it to a specified local directory.
    """
    repo_id = "MahmoodLab/conch"
    
    print(f"📥 Downloading CONCH model from Hugging Face ({repo_id})")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=os.path.join(save_dir, ".hf_cache"),
        local_dir=save_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5"]
    )
    print(f"Model saved to: {save_dir}")

if __name__ == "__main__":
    save_path = "/project/hnguyen2/mvu9/checkpoints/conch_checkpoints"
    os.makedirs(save_path, exist_ok=True)
    download_conch_model(save_path)
