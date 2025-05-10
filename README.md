# WSI pathology dataset and dataloader 
This repo include data preprocessing and dataloader for Pathology and WSIs 

## Camelyon 16 dataset: 



## TCGA dataset 
### Download
Guidance: https://andrewjanowczyk.com/download-tcga-digital-pathology-images-ffpe/

Example 

```
./gdc-client download -m gdc_manifest.txt -d path/save/data 
./gdc-client download -m manifest/LUSC/gdc_manifest.2025-05-09.192912.txt -d /project/hnguyen2/mvu9/datasets/TGCA-datasets/LUSC

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

