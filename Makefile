metadata_all: metadata_kirc metadata_kirp metadata_lusc metadata_luad
#--------SIMAE --------- 
metadata_kich_simea:
	python pipeline/tgca/generate_metadata.py --config configs_simea/data_kich.yaml

pp_kich_simea: 
	python pipeline/tgca/create_patches_fp.py --config configs_simea/data_kich.yaml  

#--------MAUI--------- 
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


#--------MAUI--------- 
get_split:
	python few-shot/tgca-renal_generate_split.py \
		--configs configs/data_kirc.yaml configs/data_kich.yaml configs/data_kirp.yaml \
		--output_dir /project/hnguyen2/mvu9/processing_datasets/tcga_renal_fewshot
