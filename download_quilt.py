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
    base_save_path = "/project/hnguyen2/mvu9/checkpoints"
    os.makedirs(base_save_path, exist_ok=True)

    # === QuiltNet-B-32 ===
    quiltnet_repo = "wisdomik/QuiltNet-B-32"
    quiltnet_dir = os.path.join(base_save_path, "QuiltNet-B-32")
    os.makedirs(quiltnet_dir, exist_ok=True)
    download_model(quiltnet_repo, quiltnet_dir)

    # # === Quilt-LLaVA-v1.5-7B ===
    # quilt_llava_repo = "wisdomik/Quilt-Llava-v1.5-7b"
    # quilt_llava_dir = os.path.join(base_save_path, "Quilt-Llava-v1.5-7b")
    # os.makedirs(quilt_llava_dir, exist_ok=True)
    # download_model(quilt_llava_repo, quilt_llava_dir)
