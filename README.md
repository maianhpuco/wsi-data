# WSI pathology dataset and dataloader 
This repo include data preprocessing and dataloader for Pathology and WSIs 

## Camelyon 16 dataset: 



## TCGA dataset 

### Download
Guidance: https://andrewjanowczyk.com/download-tcga-digital-pathology-images-ffpe/

Example 

```
./gdc-client download -m gdc_manifest.txt -d path/save/data 
./gdc-client download -m manifest/LUSC -d /project/hnguyen2/mvu9/datasets/TGCA-datasets/LUSC




```


### Fail Download Fix Command 

``` 
ls /project/hnguyen2/mvu9/datasets/TGCA-datasets/KIRP > downloaded_ids_kirp.txt
cut -f1 manifest/KIRP/gdc_manifest.2025-05-09.102009.txt | tail -n +2 > all_ids_kirp.txt
comm -23 <(sort all_ids_kirp.txt) <(sort downloaded_ids_kirp.txt) > failed_ids_kirp.txt 

cat failed_ids_kirp.txt | jq -R -s -c 'split("\n") | map(select(length > 0))' > ids_kirp.json
echo '{"ids":'$(cat ids_kirp.json)'}' > request_kirp.json

curl -s -X POST https://api.gdc.cancer.gov/manifest \
  -H "Content-Type: application/json" \
  -d @request_kirp.json \
  -o failed_manifest_kirp.txt

./gdc-client download -m failed_manifest_kirp.txt -d /project/hnguyen2/mvu9/datasets/TGCA-datasets/KIRP 
```

- simea 
chmod +x gdc-client 

./gdc-client download -m manifest/KICH/ -d ~/datasets/TGCA-datasets/KICH 
./gdc-client download -m manifest/KIRP/ -d ~/datasets/TGCA-datasets/KIRP
./gdc-client download -m manifest/LUAD/ -d ~/datasets/TGCA-datasets/LUAD
==================================Customize==================================
```
conda install -c conda-forge \
    h5py=2.10.0 \
    matplotlib=3.1.1 \
    numpy=1.18.1 \
    pandas=1.1.3 \
    pillow=7.0.0 \
    scikit-learn=0.22.1 \
    scipy=1.4.1 \
    openslide-python=1.1.1 \
    openslide=3.4.1 \
    pytorch=1.6.0 \
    torchvision=0.7.0 \
    -y

pip install \
    tensorboardx==1.9 \
    captum==0.2.0 \
    shap==0.35.0 \
    clip==1.0 \
    opencv-python==4.1.1.26

pip install opencv-python-headless==4.1.1.26

``` 

==================================Check-list================================== 
====download==== 
Compare manifest files vs data download: 
TGCA - Renal 
[x] KICH 
[x] KIRP 
[x] KIRC

TCGA-Lung 

[x] LUSD  
[x] LUAD  

====metadata generation====
TGCA - Renal 
[x] KICH 
[x] KIRP 
[x] KIRC

TCGA-Lung 
[] LUSC
[] LUAD    

====preprocessing -> h5====


TGCA - Renal 
[] KICH 
[] KIRP 
[] KIRC

TCGA-Lung 
[] LUSC
[] LUAD  

```
ls /project/hnguyen2/mvu9/datasets/TGCA-datasets/KIRC > ./check_sum/downloaded_ids_kirc.txt 
cut -f1 manifest/KIRC/gdc_manifest* | tail -n +2 > all_ids_kirc.txt
comm -23 <(sort all_ids_kirc.txt) <(sort ./check_sum/downloaded_ids_kirc.txt) > failed_ids_kirc.txt 

cat failed_ids_kirc.txt | jq -R -s -c 'split("\n") | map(select(length > 0))' > ids_kirc.json
echo '{"ids":'$(cat ids_kirc.json)'}' > request_kirc.json

curl -s -X POST https://api.gdc.cancer.gov/manifest \
  -H "Content-Type: application/json" \
  -d @request_kirc.json \
  -o failed_manifest_kirc.txt

./gdc-client download -m failed_manifest_kirc.txt -d /project/hnguyen2/mvu9/datasets/TGCA-datasets/KIRC
``` 

```
ls /project/hnguyen2/mvu9/datasets/TGCA-datasets/LUAD > ./check_sum/downloaded_ids_luad.txt 
cut -f1 manifest/LUAD/gdc_manifest* | tail -n +2 > all_ids_luad.txt
comm -23 <(sort all_ids_luad.txt) <(sort ./check_sum/downloaded_ids_luad.txt) > failed_ids_luad.txt  

```

Fold note 
--------------------------------------
 Fold 1
                             patient_id  ... label
0  a26025f2-49b3-437e-94a0-18b14067932d  ...  KICH
1  836ac1ec-6d7d-4df6-845b-461bf2c3e4ad  ...  KICH
3  b5491ed3-9e8d-4c6b-9d8d-bc9ada7b4d6f  ...  KICH

[3 rows x 3 columns]
Train: 601 samples → KICH: 72, KIRP: 195, KIRC: 334
Val:   151 samples → KICH: 26, KIRP: 40, KIRC: 85
Test:  188 samples → KICH: 23, KIRP: 65, KIRC: 100
Total: 940 / 940 complete
 Saved to: /home/mvu9/processing_datasets/processing_tcga/splits_csv/fold_1
 
patient_id, slide, label 
--------------------------------------
CAMELYON16 

----Fold 1
Train samples: 216
Val   samples: 54
Test  samples: 129
Total: 399 / 399 entries used

Preview of split DataFrame:
        train  train_label         val  val_label      test  test_label
0  normal_125            0  normal_013          0  test_001         1.0
1   tumor_098            1   tumor_063          1  test_002         1.0
2   tumor_100            1  normal_052          0  test_003         0.0
Fold 1 saved to: ./data/camelyon16_folds/fold_1.csv

======
sqlite3 /home/mvu9/datasets/glomeruli/orbit.db
.table
.schema RAW_ANNOTATION
.schema RAW_DATA_FILE 


Schema Recap:
RAW_ANNOTATION: RAW_ANNOTATION_ID, RAW_DATA_FILE_ID, RAW_ANNOTATION_TYPE, DESCRIPTION, DATA (BLOB), USER_ID, MODIFY_DATE.
RAW_DATA_FILE: RAW_DATA_FILE_ID, FILENAME, MD5, PATH, etc.
DATA is a BLOB, expected to contain geometry data (likely Polygons for glomeruli annotations).
RAW_ANNOTATION_TYPE determines the annotation type, but 0 is likely not Polygons. 


conda install -c conda-forge openjdk

export JAVA_HOME=$(dirname $(dirname $(which javac)))
