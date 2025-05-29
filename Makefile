metadata_all: metadata_kirc metadata_kirp metadata_kich

metadata_all_simea: metadata_kich_simea metadata_kirp_simea metadata_kirc_simea
pp_all_simea: pp_kirp_simea pp_kich_simea pp_kirc_simea 


# ====== Processing pipeline for KPIS dataset 
pipeline_kpis_simea: metadata_kpis_simea move_file_kpis_simea pp_kpis_simea ef_kpis_simea
move_file_kpis_simea:
	python pipeline_cls/kpis/move_file.py 
prm_kpis_simea: 
	pipeline_cls/kpis/pyramidal_processing.py 
pp_kpis_simea: #run 
	python pipeline_cls/kpis/create_patches_fp.py --config configs_simea/data_kpis.yaml 
ef_kpis_simea:
	python pipeline_cls/kpis/extract_features_fp.py --config configs_simea/data_kpis.yaml
generate_split:
	python pipeline_cls/kpis/generate_split_csv.py --config configs_simea/data_kpis.yaml


# ====== Processing pipeline for Glomeruli dataset 
# TCGA Lung LUAD: 
pp_kirc_simea:
	python pipeline_cls/tcga/create_patches_fp.py --config configs_simea/data_luad.yaml


 


#  ====== TCGA Lung LUAD ====== 
metadata_luad_simea: #done 
	python pipeline_cls/tcga/generate_metadata.py --config configs_simea/data_luad.yaml 
pyramidal_luad_simea:
	python pipeline_cls/tcga/pyramidal_processing.py --config configs_simea/data_luad.yaml
pp_luad_simea:
	python pipeline_cls/tcga/create_patches_fp.py --config configs_simea/data_luad.yaml
ef_luad_simea: 
	python pipeline_cls/tcga/extract_features_fp.py --config configs_simea/data_luad.yaml
#  ====== TCGA Lung LUSC  ====== 
metadata_lusc_simea: #done 
	python pipeline_cls/tcga/generate_metadata.py --config configs_simea/data_lusc.yaml 
pyramidal_lusc_simea:
	python pipeline_cls/tcga/pyramidal_processing.py --config configs_simea/data_lusc.yaml
pp_lusc_simea:
	python pipeline_cls/tcga/create_patches_fp.py --config configs_simea/data_lusc.yaml
ef_lusc_simea: 
	python pipeline_cls/tcga/extract_features_fp.py --config configs_simea/data_lusc.yaml

	
#--------SIMAE --------- 
# metadata_kich_simea:
# 	python pipeline_cls/tcga/generate_metadata.py --config configs_simea/data_kich.yaml
# metadata_kirp_simea:
# 	python pipeline_cls/tcga/generate_metadata.py --config configs_simea/data_kirp.yaml 
# metadata_kirc_simea:
# 	python pipeline_cls/tcga/generate_metadata.py --config configs_simea/data_kirc.yaml 

##--------SIMAE --------- PREPROCESSING  
gen_split_camelyon16:
	python pipeline_cls/camelyon16/generate_split_csv.py --config configs_simea/data_camelyon16.yaml 
gen_split_tcga_renal:
	python pipeline_cls/tcga/generate_split_csv_renal.py --config configs_simea/data_tcga_renal.yaml 
gen_split_tcga_lung:
	python pipeline_cls/tcga/generate_split_csv_lung.py --config configs_simea/data_tcga_lung.yaml 


#--------SIMAE --------- PREPROCESSING 
pp_camelyon16_simea:
	python pipeline_cls/camelyon16/create_patches_fp.py --config configs_simea/data_camelyon16.yaml
pp_kich_simea: 
	python pipeline_cls/tcga/create_patches_fp.py --config configs_simea/data_kich.yaml 
pp_kirp_simea:
	python pipeline_cls/tcga/create_patches_fp.py --config configs_simea/data_kirp.yaml  
pp_kirc_simea:
	python pipeline_cls/tcga/create_patches_fp.py --config configs_simea/data_kirc.yaml

# glomeruli data need to process get multiple magnification level before create patches 
pyramidal_pp_glomeruli_simea:
	python pipeline_cls/glomeruli/pyramidal_processing.py 
pp_glomeruli_simea:
	python pipeline_cls/glomeruli/create_patches_fp.py --config configs_simea/data_glomeruli.yaml	

#====== 
# check_missing_file:
# 	python pipeline_cls/tcga/count_missing_file.py
# rerun_pp_kich_simea: 
# 	python pipeline_cls/tcga/create_patches_fp.py --config configs_simea/data_kich.yaml --csv_filenames yes
# rerun_pp_kirp_simea: 
# 	python pipeline_cls/tcga/create_patches_fp.py --config configs_simea/data_kirp.yaml --csv_filenames yes 

#--------SIMAE --------- FAST PROCESSING, H5 FEATURES | GENERATION  
ef_camelyon16_simea: 
	python pipeline_cls/camelyon16/extract_features_fp.py --config configs_simea/data_camelyon16.yaml
ef_kich_simea: 
	python pipeline_cls/tcga/extract_features_fp.py --config configs_simea/data_kich.yaml
ef_kirc_simea: 
	python pipeline_cls/tcga/extract_features_fp.py --config configs_simea/data_kirc.yaml
ef_kirp_simea: 
	python pipeline_cls/tcga/extract_features_fp.py --config configs_simea/data_kirp.yaml
ef_glomeruli_simea: 
	python pipeline_cls/glomeruli/extract_features_fp.py --config configs_simea/data_glomeruli.yaml

#--------SIMAE --------- PATCHES GENERATION 
gen_patches_kich_simea:
	python feature_extraction/generate_patches.py --config configs_simea/data_kich.yaml
gen_patches_kirc_simea:
	python feature_extraction/generate_patches.py --config configs_simea/data_kirc.yaml
gen_patches_kirp_simea:
	python feature_extraction/generate_patches.py --config configs_simea/data_kirp.yaml 
gen_patches_camelyon16_simea:
	# python pipeline_cls/camelyon16/generate_patches.py --config configs_simea/data_camelyon16.yaml
	python pipeline_cls/camelyon16/create_and_generate_patches.py --config configs_simea/data_camelyon16.yaml
 
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
	python pipeline_cls/camelyon16/anno_processing.py \
	--config configs_simea/data_camelyon16.yaml

#---------fixing / sanity check 
check_downscale_kpi:
	python pipeline_cls/kpis/sanity_check_downscale.py \
	--config configs_simea/data_kpis.yaml 

check_downscale_glo:
	python pipeline_cls/glomeruli/sanity_check_downscale.py \
	--config configs_simea/data_glomeruli.yaml
 
extract_glomeruli_masks:
	python pipeline_cls/glomeruli/extract_anno.py \
	--config configs_simea/data_glomeruli.yaml




#FEW SHOT SETTING 
# =========preprocessing========= 



#======================================FEW SHOT LEARNING ========================================== 
patch_gen_camelyon16:
	python pipeline_fewshot/camelyon16/patch_generation.py \
	--config configs_simea/data_camelyon16.yaml	--patch_size 256 --magnification 10x  
 

#TCGA 
create_csv_lung:
	python pipeline_fewshot/tcga/create_csv_lung.py 

create_csv_renal:
	python pipeline_fewshot/tcga/create_csv_renal.py 

# KICH patch generation
patch_gen_kich:
	python pipeline_fewshot/tcga/patch_generation.py \
	--config configs_simea/data_kich.yaml --patch_size 256 --magnification 10x  
 
# KICH patch generation
patch_gen_kich:
	python pipeline_fewshot/tcga/patch_generation.py \
	--config configs_simea/data_kich.yaml --patch_size 256 --magnification 10x

# KIRP patch generation
patch_gen_kirp:
	python pipeline_fewshot/tcga/patch_generation.py \
	--config configs_simea/data_kirp.yaml --patch_size 256 --magnification 10x

# KIRC patch generation
patch_gen_kirc:
	python pipeline_fewshot/tcga/patch_generation.py \
	--config configs_simea/data_kirc.yaml --patch_size 256 --magnification 10x

# LUAD patch generation
patch_gen_luad:
	python pipeline_fewshot/tcga/patch_generation.py \
	--config configs_simea/data_luad.yaml --patch_size 256 --magnification 10x

# LUSC patch generation
patch_gen_lusc:
	python pipeline_fewshot/tcga/patch_generation.py \
	--config configs_simea/data_lusc.yaml --patch_size 256 --magnification 10x
