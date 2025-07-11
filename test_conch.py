import os
import sys
import torch
import numpy as np
from PIL import Image

# Add path to CONCH submodule
sys.path.append("src/externals/CONCH")

from conch.open_clip_custom import create_model_from_pretrained

os.environ['HF_HOME'] = '/project/hnguyen2/mvu9/folder_04_ma/wsi-data/cache_folder/.cache/huggingface'
 
def main():
    # Load Hugging Face token from environment
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        raise ValueError("Please set HF_TOKEN in your environment.")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CONCH model and preprocessing
    model, preprocess = create_model_from_pretrained(
        'conch_ViT-B-16',
        "hf_hub:MahmoodLab/conch",
        hf_auth_token=hf_token
    )
    model = model.to(device)
    model.eval()

    # Create a random RGB image of size 224x224
    random_array = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(random_array)

    # Preprocess and encode
    image_tensor = preprocess(image).unsqueeze(0).to(device)  # shape: [1, 3, 224, 224]

    with torch.inference_mode():
        features = model.encode_image(image_tensor, proj_contrast=True, normalize=True)

    print("Image encoded.")
    print("Feature vector shape:", features.shape)
    print("First 5 values:", features[0][:5])

if __name__ == "__main__":
    main()

 