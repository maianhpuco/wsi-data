metadata_all: metadata_kirc metadata_kirp metadata_lusc metadata_luad

metadata_kich:
	python scripts/preprocessing/tcga/generate_metadata.py --config configs/data_tcga_kich.yaml
metadata_kirc:
	python scripts/preprocessing/tcga/generate_metadata.py --config configs/data_tcga_kirc.yaml
metadata_kirp:
	python scripts/preprocessing/tcga/generate_metadata.py --config configs/data_tcga_kirp.yaml
metadata_lusc:
	python scripts/preprocessing/tcga/generate_metadata.py --config configs/data_tcga_lusc.yaml
metadata_luad:
	python scripts/preprocessing/tcga/generate_metadata.py --config configs/data_tcga_luad.yaml

pp_kich: 
	python create_patches_fp.py --config configs/data_kich.yaml 
