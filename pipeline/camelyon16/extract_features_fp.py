import os
import sys
import argparse
import yaml
import subprocess
import torch

# Ensure CLAM is in the import path
sys.path.append("src/externals/CLAM")

# Check CUDA availability
if torch.cuda.is_available():
    print(f"✅ CUDA is available. Using: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ CUDA is NOT available. Using CPU only.")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Set paths from config
    source = cfg['paths']['source']
    patch_h5_dir = cfg['paths']['patch_h5_dir']
    feat_dir = os.path.join(cfg['paths']['save_dir'], 'features_fp')
    csv_path = os.path.join(cfg['paths']['save_dir'], 'slide_list.csv')

    # Create necessary directories
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(os.path.join(feat_dir, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(feat_dir, 'h5_files'), exist_ok=True)

    # Generate slide list if not found
    # if not os.path.exists(csv_path):
    #     print(f"🔧 Generating slide list CSV at: {csv_path}")
    #     slide_ext = cfg.get("feature_extraction", {}).get("slide_ext", ".tif")
    #     slide_files = [f for f in os.listdir(source) if f.endswith(slide_ext)]
    #     # with open(csv_path, 'w') as f:
    #     #     for s in slide_files:
    #     #         f.write(s + '\n')
    #     with open(csv_path, 'w') as f:
    #         f.write("slide_id\n")
    #         for s in slide_files:
    #             f.write(s + '\n')
    if not os.path.exists(csv_path):
        print(f"🔧 Generating slide list CSV at: {csv_path}")
        slide_ext = cfg.get("feature_extraction", {}).get("slide_ext", ".tif")
        slide_files = [f for f in os.listdir(source) if f.endswith(slide_ext)]
        if not slide_files:
            print(f"❌ No slides found in {source} with extension {slide_ext}")
            sys.exit(1)
        print(f"Found {len(slide_files)} slides: {slide_files[:5]}")  # Print first 5 for debugging
        with open(csv_path, 'w') as f:
            f.write("slide_id\n")
            for s in slide_files:
                f.write(s + '\n')  
    # Feature extraction config
    feat_cfg = cfg.get("feature_extraction", {})
    model_name = feat_cfg.get("model_name", "resnet50_trunc")
    batch_size = feat_cfg.get("batch_size", 256)
    target_patch_size = feat_cfg.get("target_patch_size", 224)
    slide_ext = feat_cfg.get("slide_ext", ".tif")
    no_auto_skip = feat_cfg.get("no_auto_skip", False)

    # Assemble command
    cmd = [
        "python", "src/externals/CLAM/extract_features_fp.py",
        "--data_h5_dir", patch_h5_dir,
        "--data_slide_dir", source,
        "--slide_ext", slide_ext,
        "--csv_path", csv_path,
        "--feat_dir", feat_dir,
        "--model_name", model_name,
        "--batch_size", str(batch_size),
        "--target_patch_size", str(target_patch_size),
    ]
    if no_auto_skip:
        cmd.append("--no_auto_skip")

    print("🚀 Running feature extraction command:")
    print(" ".join(cmd))

    # Run CLAM's script
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
