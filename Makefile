metadata_all: metadata_kirc metadata_kirp metadata_kich


metadata_all_simea: metadata_kich_simea metadata_kirp_simea metadata_kirc_simea
pp_all_simea: pp_kirp_simea pp_kich_simea pp_kirc_simea 


#--------SIMAE --------- 
# metadata_kich_simea:
# 	python pipeline/tcga/generate_metadata.py --config configs_simea/data_kich.yaml
# metadata_kirp_simea:
# 	python pipeline/tcga/generate_metadata.py --config configs_simea/data_kirp.yaml 
# metadata_kirc_simea:
# 	python pipeline/tcga/generate_metadata.py --config configs_simea/data_kirc.yaml 

##--------SIMAE --------- PREPROCESSING  
gen_split_camelyon16:
	python pipeline/camelyon16/generate_split_csv.py --config configs_simea/data_camelyon16.yaml 
gen_split_tcga_renal:
	python pipeline/tcga/generate_split_csv.py --config configs_simea/data_tcga_renal.yaml 


#--------SIMAE --------- PREPROCESSING 
pp_camelyon16_simea:
	python pipeline/camelyon16/create_patches_fp.py --config configs_simea/data_camelyon16.yaml

pp_kich_simea: 
	python pipeline/tcga/create_patches_fp.py --config configs_simea/data_kich.yaml 
pp_kirp_simea:
	python pipeline/tcga/create_patches_fp.py --config configs_simea/data_kirp.yaml  
pp_kirc_simea:
	python pipeline/tcga/create_patches_fp.py --config configs_simea/data_kirc.yaml
	 
#====== 
# check_missing_file:
# 	python pipeline/tcga/count_missing_file.py
# rerun_pp_kich_simea: 
# 	python pipeline/tcga/create_patches_fp.py --config configs_simea/data_kich.yaml --csv_filenames yes
# rerun_pp_kirp_simea: 
# 	python pipeline/tcga/create_patches_fp.py --config configs_simea/data_kirp.yaml --csv_filenames yes 

#--------SIMAE --------- FAST PROCESSING, H5 FEATURES | GENERATION  
ef_camelyon16_simea: 
	python pipeline/camelyon16/extract_features_fp.py --config configs_simea/data_camelyon16.yaml
ef_kich_simea: 
	python pipeline/tcga/extract_features_fp.py --config configs_simea/data_kich.yaml
ef_kirc_simea: 
	python pipeline/tcga/extract_features_fp.py --config configs_simea/data_kirc.yaml
ef_kirp_simea: 
	python pipeline/tcga/extract_features_fp.py --config configs_simea/data_kirp.yaml

#--------SIMAE --------- PATCHES GENERATION 
gen_patches_kich_simea:
	python feature_extraction/generate_patches.py --config configs_simea/data_kich.yaml
gen_patches_kirc_simea:
	python feature_extraction/generate_patches.py --config configs_simea/data_kirc.yaml
gen_patches_kirp_simea:
	python feature_extraction/generate_patches.py --config configs_simea/data_kirp.yaml 
gen_patches_camelyon16_simea:
	# python pipeline/camelyon16/generate_patches.py --config configs_simea/data_camelyon16.yaml
	python pipeline/camelyon16/create_and_generate_patches.py --config configs_simea/data_camelyon16.yaml
 
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
	python few-shot/tcga-renal_generate_split.py \
		--configs configs/data_kirc.yaml configs/data_kich.yaml configs/data_kirp.yaml \
		--output_dir /project/hnguyen2/mvu9/processing_datasets/tcga_renal_fewshot

cam_anno_process:
	python pipeline/camelyon16/anno_processing.py \
	--config configs_simea/data_camelyon16.yaml

#---------fixing / sanity check 
check_downscale_kpi:
	python pipeline/kpis/sanity_check_downscale.py \
	--config configs_simea/data_kpis.yaml 

check_downscale_glo:
	python pipeline/kpis/sanity_check_downscale.py \
	--config config_glomeruli.yaml
 