import openslide

slide = openslide.OpenSlide("/home/mvu9/processing_datasets/glomeruli_pyramidal/train/6654588.tiff")
print("Level count:", slide.level_count)
print("Dimensions at each level:", slide.level_dimensions)
