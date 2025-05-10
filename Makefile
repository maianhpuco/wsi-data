all: metadata_kich

metadata_kich:
	python scripts/preprocessing/tcga/generate_metadata.py --config configs/data_tcga_kich.yaml

pp_kich: 
	python create_patches_fp.py --config configs/data_kich.yaml 
	