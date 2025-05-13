metadata_all: metadata_kirc metadata_kirp metadata_kich



metadata_all_simea: metadata_kich_simea metadata_kirp_simea metadata_kirc_simea
pp_all_simea: pp_kirp_simea pp_kirc_simea pp_kirc_simea 


#--------SIMAE --------- 
metadata_kich_simea:
	python pipeline/tgca/generate_metadata.py --config configs_simea/data_kich.yaml
metadata_kirp_simea:
	python pipeline/tgca/generate_metadata.py --config configs_simea/data_kirp.yaml 
metadata_kirc_simea:
	python pipeline/tgca/generate_metadata.py --config configs_simea/data_kirc.yaml 




#--------SIMAE --------- PREPROCESSING 
pp_camelyon16_simea:
	python pipeline/camelyon16/create_patches_fp.py --config configs_simea/data_camelyon16.yaml

pp_kich_simea: 
	python pipeline/tgca/create_patches_fp.py --config configs_simea/data_kich.yaml 

pp_kirp_simea:
	python pipeline/tgca/create_patches_fp.py --config configs_simea/data_kirp.yaml  

pp_kirc_simea:
	python pipeline/tgca/create_patches_fp.py --config configs_simea/data_kirc.yaml  

#--------SIMAE --------- PATCHES GENERATION 
gen_patches_kich_simea:
	python feature_extraction/generate_patches.py --config configs_simea/data_kich.yaml

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
