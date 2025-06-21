import os
from huggingface_hub import snapshot_download

def download_model(repo_id, save_dir):
    """
    Downloads a Hugging Face model and saves it to a local directory.
    """
    print(f"Downloading: {repo_id}")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=os.path.join(save_dir, ".hf_cache"),
        local_dir=save_dir,
        local_dir_use_symlinks=False
    )
    print(f"Saved to: {save_dir}\n")

if __name__ == "__main__":
    base_save_path = "/project/hnguyen2/mvu9/pretrained_checkpoints"
    os.makedirs(base_save_path, exist_ok=True)
    # === PLIP ===
    plip_repo = "vinid/plip"
    plip_dir = os.path.join(base_save_path, "PLIP")
    os.makedirs(plip_dir, exist_ok=True)
    download_model(plip_repo, plip_dir)
