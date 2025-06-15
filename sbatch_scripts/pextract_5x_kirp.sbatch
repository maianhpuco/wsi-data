#!/bin/bash

#SBATCH --job-name=pextract_5x_kirp
#SBATCH --output=logs/pextract_5x_kirp.log
#SBATCH --nodelist=compute-0-2
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00  # 1 day

echo "Starting pextract_5x_kirp job on compute-0-2"

python pipeline_fewshot/tcga/patch_extraction.py --config configs_maui/data_kirp.yaml --patch_size 256 --magnification 10x   

echo "Done"

# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kirc/features_fp/pt_files/TCGA-CZ-4864-01Z-00-DX1.4c3b083b-cfd3-4b75-9a5f-49286af1b433.pt ./kirc/features_fp/pt_files/
# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kirc/features_fp/h5_files/TCGA-CZ-4864-01Z-00-DX1.4c3b083b-cfd3-4b75-9a5f-49286af1b433.h5 ./kirc/features_fp/h5_files/

# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kirc/features_fp/pt_files/TCGA-T7-A92I-01Z-00-DX3.FC717D92-13BD-4968-8E82-0CC53EB6D5D1.pt ./kirc/features_fp/pt_files/
# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kirc/features_fp/h5_files/TCGA-T7-A92I-01Z-00-DX3.FC717D92-13BD-4968-8E82-0CC53EB6D5D1.h5 ./kirc/features_fp/h5_files/

# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kirc/features_fp/pt_files/TCGA-B8-5159-01Z-00-DX1.12a9fb4e-2062-4e69-bd04-61eb2c1a7501.pt ./kirc/features_fp/pt_files/
# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kirc/features_fp/h5_files/TCGA-B8-5159-01Z-00-DX1.12a9fb4e-2062-4e69-bd04-61eb2c1a7501.h5 ./kirc/features_fp/h5_files/

# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kirc/features_fp/pt_files/TCGA-CZ-4856-01Z-00-DX1.866ea89f-a487-4380-af9b-bec919edeefb.pt ./kirc/features_fp/pt_files/
# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kirc/features_fp/h5_files/TCGA-CZ-4856-01Z-00-DX1.866ea89f-a487-4380-af9b-bec919edeefb.h5 ./kirc/features_fp/h5_files/

# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kirc/features_fp/pt_files/TCGA-CZ-4859-01Z-00-DX1.ef46368c-c76e-4fc9-a620-65b7f63fb083.pt ./kirc/features_fp/pt_files/
# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kirc/features_fp/h5_files/TCGA-CZ-4859-01Z-00-DX1.ef46368c-c76e-4fc9-a620-65b7f63fb083.h5 ./kirc/features_fp/h5_files/








# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kich/features_fp/pt_files/TCGA-KL-8335-01Z-00-DX1.3F291515-CD98-4D8F-A255-A10A69139242.pt ./kich/features_fp/pt_files/
# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kich/features_fp/h5_files/TCGA-KL-8335-01Z-00-DX1.3F291515-CD98-4D8F-A255-A10A69139242.h5 ./kich/features_fp/h5_files/






# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kirp/features_fp/pt_files/TCGA-HE-A5NK-01Z-00-DX1.E2D22525-AF0E-4CC1-9893-FB1FB80800C3.pt ./kirp/features_fp/pt_files/
# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kirp/features_fp/h5_files/TCGA-HE-A5NK-01Z-00-DX1.E2D22525-AF0E-4CC1-9893-FB1FB80800C3.h5 ./kirp/features_fp/h5_files/

# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kirp/features_fp/pt_files/TCGA-IA-A83T-01Z-00-DX1.12419A50-9BAC-4469-AF8B-546BB6947B8D.pt ./kirp/features_fp/pt_files/
# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kirp/features_fp/h5_files/TCGA-IA-A83T-01Z-00-DX1.12419A50-9BAC-4469-AF8B-546BB6947B8D.h5 ./kirp/features_fp/h5_files/

# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kirp/features_fp/pt_files/TCGA-2Z-A9JE-01Z-00-DX1.B2538A85-D0EA-4C50-BD07-1C59DD8208F6.pt ./kirp/features_fp/pt_files/
# scp -r mvu9@simea.ee.e.uh.edu:/home/mvu9/processing_datasets/processing_tcga_256/kirp/features_fp/h5_files/TCGA-2Z-A9JE-01Z-00-DX1.B2538A85-D0EA-4C50-BD07-1C59DD8208F6.h5 ./kirp/features_fp/h5_files/