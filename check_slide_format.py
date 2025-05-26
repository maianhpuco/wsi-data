import openslide
import os

slide_dir = "/home/mvu9/processing_datasets/glomeruli_pyramidal/train"

tiff_files = [f for f in os.listdir(slide_dir) if f.endswith(".tif") or f.endswith(".tiff")]
print(f"🔍 Found {len(tiff_files)} TIFF files in {slide_dir}")

for filename in sorted(tiff_files):
    slide_path = os.path.join(slide_dir, filename)
    try:
        slide = openslide.OpenSlide(slide_path)
        print(f"\n {filename}")
        print(f"  Level count: {slide.level_count}")
        print(f"  Dimensions at each level: {slide.level_dimensions}")
        if slide.level_count < 5:
            print(f"  Warning: {filename} has less than 5 levels.") 
        if slide.level_count > 5:
            print(f"  Warning: {filename} has more than 5 levels.") 
            break  
    except Exception as e:
        print(f"===> Failed to open {filename}: {e}")
