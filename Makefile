
#------------------- simea processing pipeline ------------------- 
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
gen_split_camelyon16_simea:
	python pipeline_cls/camelyon16/generate_split_csv.py --config configs_simea/data_camelyon16.yaml 
gen_split_tcga_renal_simea:
	python pipeline_cls/tcga/generate_split_csv_renal.py --config configs_simea/data_tcga_renal.yaml 
gen_split_tcga_lung_simea:
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


#------------------- maui processing pipeline -------------------
# --- generate metadata for TCGA datasets:  no need sbatch 
tcga_metadata_maui: metadata_kich_maui metadata_kirp_maui metadata_kirc_maui metadata_luad_maui metadata_lusc_maui 
metadata_kich_maui:
	python pipeline_cls/tcga/generate_metadata.py --config configs_maui/data_kich.yaml
metadata_kirp_maui:
	python pipeline_cls/tcga/generate_metadata.py --config configs_maui/data_kirp.yaml 
metadata_kirc_maui:
	python pipeline_cls/tcga/generate_metadata.py --config configs_maui/data_kirc.yaml 
metadata_luad_maui:
	python pipeline_cls/tcga/generate_metadata.py --config configs_maui/data_luad.yaml 
metadata_lusc_maui:
	python pipeline_cls/tcga/generate_metadata.py --config configs_maui/data_lusc.yaml 



#--------PREPROCESSING 
sb_pp_camelyon16_maui:
	sbatch ./sbatch_scripts/pp_camelyon16_maui.sbatch 
sb_pp_kich_maui:
	sbatch ./sbatch_scripts/pp_kich_maui.sbatch
sb_pp_kirc_maui:
	sbatch ./sbatch_scripts/pp_kirc_maui.sbatch
sb_pp_kirp_maui:
	sbatch ./sbatch_scripts/pp_kirp_maui.sbatch
sb_pp_luad_maui:
	sbatch ./sbatch_scripts/pp_luad_maui.sbatch
sb_pp_lusc_maui:
	sbatch ./sbatch_scripts/pp_lusc_maui.sbatch	


sb_ef_camelyon16_maui:
	sbatch ./sbatch_scripts/ef_camelyon16_maui.sbatch
sb_ef_kich_maui:
	sbatch ./sbatch_scripts/ef_kich_maui.sbatch
sb_ef_kirc_maui:
	sbatch ./sbatch_scripts/ef_kirc_maui.sbatch
sb_ef_kirp_maui:
	sbatch ./sbatch_scripts/ef_kirp_maui.sbatch
sb_ef_luad_maui:
	sbatch ./sbatch_scripts/ef_luad_maui.sbatch
sb_ef_lusc_maui:
	sbatch ./sbatch_scripts/ef_lusc_maui.sbatch

##--------GEN SPLIT 
gen_split_camelyon16_maui:
	python pipeline_cls/camelyon16/generate_split_csv.py --config configs_maui/data_camelyon16.yaml 
gen_split_tcga_renal_maui:
	python pipeline_cls/tcga/generate_split_csv_renal.py --config configs_maui/data_tcga_renal.yaml 
gen_split_tcga_lung_maui:
	python pipeline_cls/tcga/generate_split_csv_lung.py --config configs_maui/data_tcga_lung.yaml 




#---------------------------------
#--------SIMAE --------- PATCHES GENERATION 
# ==== Patch Generation Jobs (5x) ====
pg_5x_kich:
	sbatch sbatch_scripts/pg_5x_kich.sbatch
pg_5x_kirp:
	sbatch sbatch_scripts/pg_5x_kirp.sbatch
pg_5x_kirc:
	sbatch sbatch_scripts/pg_5x_kirc.sbatch
pg_5x_luad:
	sbatch sbatch_scripts/pg_5x_luad.sbatch
pg_5x_lusc:
	sbatch sbatch_scripts/pg_5x_lusc.sbatch

pg_all_5x: \
	pg_5x_kich \
	pg_5x_kirp \
	pg_5x_kirc \
	pg_5x_luad \
	pg_5x_lusc
 
# ==== Patch Generation Jobs (10x) ====
pg_10x_kich:
	sbatch sbatch_scripts/pg_10x_kich.sbatch
pg_10x_kirp:
	sbatch sbatch_scripts/pg_10x_kirp.sbatch
pg_10x_kirc:
	sbatch sbatch_scripts/pg_10x_kirc.sbatch
pg_10x_luad:
	sbatch sbatch_scripts/pg_10x_luad.sbatch
pg_10x_lusc:
	sbatch sbatch_scripts/pg_10x_lusc.sbatch

pg_all_10x: \
	pg_10x_kich \
	pg_10x_kirp \
	pg_10x_kirc \
	pg_10x_luad \
	pg_10x_lusc
 


#------------------------------------------

# ==== Prototype Generation Jobs (20x) ====
pg_kich_20x:
	sbatch sbatch_scripts/pg_20x_kich.sbatch
pg_kirp_20x:
	sbatch sbatch_scripts/pg_20x_kirp.sbatch
pg_kirc_20x:
	sbatch sbatch_scripts/pg_20x_kirc.sbatch
pg_luad_20x:
	sbatch sbatch_scripts/pg_20x_luad.sbatch
pg_lusc_20x:
	sbatch sbatch_scripts/pg_20x_lusc.sbatch

pg_all_20x: \
	pg_kich_20x \
	pg_kirp_20x \
	pg_kirc_20x \
	pg_luad_20x \
	pg_lusc_20x


# ==== Prototype Generation Jobs (40x) ====
pg_kich_40x:
	sbatch sbatch_scripts/pg_40x_kich.sbatch
pg_kirp_40x:
	sbatch sbatch_scripts/pg_40x_kirp.sbatch
pg_kirc_40x:
	sbatch sbatch_scripts/pg_40x_kirc.sbatch
pg_luad_40x:
	sbatch sbatch_scripts/pg_40x_luad.sbatch
pg_lusc_40x:
	sbatch sbatch_scripts/pg_40x_lusc.sbatch

pg_all_40x: \
	pg_kich_40x \
	pg_kirp_40x \
	pg_kirc_40x \
	pg_luad_40x \
	pg_lusc_40x


# ==== Run All Prototype Generation Jobs ====
pg_all: pg_all_20x pg_all_40x


sampling_kich: 
	sbatch sbatch_scripts/s_kich.sbatch
sampling_kirp: 
	sbatch sbatch_scripts/s_kirp.sbatch 
sampling_kirc:
	sbatch sbatch_scripts/s_kirc.sbatch 
sampling_luad:
	sbatch sbatch_scripts/s_luad.sbatch  
sampling_lusc:
	sbatch sbatch_scripts/s_lusc.sbatch

sampling: sampling_kich sampling_kirc sampling_kirp sampling_luad sampling_lusc
# sampling_lung: sampling_luad sampling_lusc

sampling_h5pt_kich:
	sbatch sbatch_scripts/s_h5pt_kich.sbatch

sampling_h5pt_kirp:
	sbatch sbatch_scripts/s_h5pt_kirp.sbatch

sampling_h5pt_kirc:
	sbatch sbatch_scripts/s_h5pt_kirc.sbatch

sampling_h5pt_luad:
	sbatch sbatch_scripts/s_h5pt_luad.sbatch

sampling_h5pt_lusc:
	sbatch sbatch_scripts/s_h5pt_lusc.sbatch


sampling_h5pt: sampling_h5pt_kich sampling_h5pt_kirp sampling_h5pt_kirc  #sampling_h5pt_luad sampling_h5pt_lusc

#------------------------------------------ 
# GENERATE FEATURE FROM CLIP RN 50 

# ==== Sampling h5 and pt files ====
sampling_h5pt_kich:
	sbatch sbatch_scripts/s_h5pt_kich.sbatch

sampling_h5pt_kirp:
	sbatch sbatch_scripts/s_h5pt_kirp.sbatch

sampling_h5pt_kirc:
	sbatch sbatch_scripts/s_h5pt_kirc.sbatch

sampling_h5pt_luad:
	sbatch sbatch_scripts/s_h5pt_luad.sbatch

sampling_h5pt_lusc:
	sbatch sbatch_scripts/s_h5pt_lusc.sbatch

sampling_h5pt: sampling_h5pt_kich sampling_h5pt_kirp sampling_h5pt_kirc  #sampling_h5pt_luad sampling_h5pt_lusc


# ==== Patch extraction jobs (10x) ====
pextract_kich_10x:
	sbatch sbatch_scripts/pextract_10x_kich.sbatch
pextract_kirp_10x:
	sbatch sbatch_scripts/pextract_10x_kirp.sbatch
pextract_kirc_10x:
	sbatch sbatch_scripts/pextract_10x_kirc.sbatch
pextract_luad_10x:
	sbatch sbatch_scripts/pextract_10x_luad.sbatch
pextract_lusc_10x:
	sbatch sbatch_scripts/pextract_10x_lusc.sbatch
pextract_all_10x: pextract_kich_10x pextract_kirp_10x pextract_kirc_10x pextract_luad_10x pextract_lusc_10x


pextract_kich_5x:
	sbatch sbatch_scripts/pextract_5x_kich.sbatch
pextract_kirp_5x:
	sbatch sbatch_scripts/pextract_5x_kirp.sbatch
pextract_kirc_5x:
	sbatch sbatch_scripts/pextract_5x_kirc.sbatch
pextract_luad_5x:
	sbatch sbatch_scripts/pextract_5x_luad.sbatch
pextract_lusc_5x:
	sbatch sbatch_scripts/pextract_5x_lusc.sbatch
pextract_all_5x: pextract_kich_5x pextract_kirp_5x pextract_kirc_5x pextract_luad_5x pextract_lusc_5x


#------- 
# ==== Patch extraction jobs using CONCH (10x) ====
pextract_conch_kich_10x:
	sbatch sbatch_scripts/pextract_conch_10x_kich.sbatch
pextract_conch_kirp_10x:
	sbatch sbatch_scripts/pextract_conch_10x_kirp.sbatch
pextract_conch_kirc_10x:
	sbatch sbatch_scripts/pextract_conch_10x_kirc.sbatch
pextract_conch_luad_10x:
	sbatch sbatch_scripts/pextract_conch_10x_luad.sbatch
pextract_conch_lusc_10x:
	sbatch sbatch_scripts/pextract_conch_10x_lusc.sbatch

pextract_conch_all_10x: \
	pextract_conch_kich_10x \
	pextract_conch_kirp_10x \
	pextract_conch_kirc_10x \
	pextract_conch_luad_10x \
	pextract_conch_lusc_10x


# ==== Patch extraction jobs using CONCH (5x) ====
pextract_conch_kich_5x:
	sbatch sbatch_scripts/pextract_conch_5x_kich.sbatch
pextract_conch_kirp_5x:
	sbatch sbatch_scripts/pextract_conch_5x_kirp.sbatch
pextract_conch_kirc_5x:
	sbatch sbatch_scripts/pextract_conch_5x_kirc.sbatch
pextract_conch_luad_5x:
	sbatch sbatch_scripts/pextract_conch_5x_luad.sbatch
pextract_conch_lusc_5x:
	sbatch sbatch_scripts/pextract_conch_5x_lusc.sbatch

pextract_conch_all_5x: \
	pextract_conch_kich_5x \
	pextract_conch_kirp_5x \
	pextract_conch_kirc_5x \
	pextract_conch_luad_5x \
	pextract_conch_lusc_5x

# ==== Patch extraction jobs using QUILT (10x) ====
pextract_quilt_kich_10x:
	sbatch sbatch_scripts/pextract_quilt_10x_kich.sbatch
pextract_quilt_kirp_10x:
	sbatch sbatch_scripts/pextract_quilt_10x_kirp.sbatch
pextract_quilt_kirc_10x:
	sbatch sbatch_scripts/pextract_quilt_10x_kirc.sbatch
pextract_quilt_luad_10x:
	sbatch sbatch_scripts/pextract_quilt_10x_luad.sbatch
pextract_quilt_lusc_10x:
	sbatch sbatch_scripts/pextract_quilt_10x_lusc.sbatch

pextract_quilt_all_10x: \
	pextract_quilt_kich_10x \
	pextract_quilt_kirp_10x \
	pextract_quilt_kirc_10x \
	pextract_quilt_luad_10x \
	pextract_quilt_lusc_10x

# ==== Patch extraction jobs using QUILT (5x) ====
pextract_quilt_kich_5x:
	sbatch sbatch_scripts/pextract_quilt_5x_kich.sbatch
pextract_quilt_kirp_5x:
	sbatch sbatch_scripts/pextract_quilt_5x_kirp.sbatch
pextract_quilt_kirc_5x:
	sbatch sbatch_scripts/pextract_quilt_5x_kirc.sbatch
pextract_quilt_luad_5x:
	sbatch sbatch_scripts/pextract_quilt_5x_luad.sbatch
pextract_quilt_lusc_5x:
	sbatch sbatch_scripts/pextract_quilt_5x_lusc.sbatch

pextract_quilt_all_5x: \
	pextract_quilt_kich_5x \
	pextract_quilt_kirp_5x \
	pextract_quilt_kirc_5x \
	pextract_quilt_luad_5x \
	pextract_quilt_lusc_5x


# === Root Paths ===
BASE_DIR = /project/hnguyen2/mvu9/processing_datasets/processing_tcga_256

# === Feature Types ===
FEATURE_TYPES = clip_rn50_features_fp conch_features_fp quilt_features_fp

# === Cancer Types ===
CANCERS = kich kirc kirp

# === Magnifications ===
MAGS = 5x 10x

# === Zip Commands for Each Combination ===
$(foreach feature,$(FEATURE_TYPES),\
  $(foreach cancer,$(CANCERS),\
    $(foreach mag,$(MAGS),\
      $(eval zip_$(feature)_$(cancer)_$(mag): ; \
        zip -rq $(cancer)_$(feature)_$(mag).zip $(BASE_DIR)/$(cancer)/$(feature)/patch_256x256_$(mag))\
  )))

# === Grouped Targets ===

# Zip all for a specific cancer
zip_all_kich: \
  zip_clip_rn50_features_fp_kich_5x \
  zip_clip_rn50_features_fp_kich_10x \
  zip_conch_features_fp_kich_5x \
  zip_conch_features_fp_kich_10x \
  zip_quilt_features_fp_kich_5x \
  zip_quilt_features_fp_kich_10x

zip_all_kirc: \
  zip_clip_rn50_features_fp_kirc_5x \
  zip_clip_rn50_features_fp_kirc_10x \
  zip_conch_features_fp_kirc_5x \
  zip_conch_features_fp_kirc_10x \
  zip_quilt_features_fp_kirc_5x \
  zip_quilt_features_fp_kirc_10x

zip_all_kirp: \
  zip_clip_rn50_features_fp_kirp_5x \
  zip_clip_rn50_features_fp_kirp_10x \
  zip_conch_features_fp_kirp_5x \
  zip_conch_features_fp_kirp_10x \
  zip_quilt_features_fp_kirp_5x \
  zip_quilt_features_fp_kirp_10x

# Zip everything
zip_all: zip_all_kich zip_all_kirc zip_all_kirp


######### 
check_data_conch_5x:
	python check_data.py --config configs_maui/data_tcga_renal.yaml --data_dir_map conch_patch_256x256_5x --k_start 1 --k_end 1

