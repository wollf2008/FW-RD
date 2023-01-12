# SLFCD
## Preprocess
### Annotations
xml annotation to json
```shell
python preprocess/wsi/bin/camelyon16xml2json.py {original_dataset/camelyon16/testing/lesion_annotations/} {preprocess/jsons/test/}
```
### Generate image patches
1. Generate tissue masks for WSIs
```shell
python preprocess/wsi/bin/tissue_mask.py original_dataset/camelyon16/testing/images original_dataset/camelyon16/testing/images_tissue
```
2. Generate tumor masks for WSIs
```shell
python preprocess/wsi/bin/tumor_mask.py original_dataset/camelyon16/testing/images preprocess/jsons/test/ original_dataset/camelyon16/testing/images_tumor
```
3. Generate masks for normal areas
```shell
python preprocess/wsi/bin/non_tumor_mask.py original_dataset/camelyon16/testing/images_tumor original_dataset/camelyon16/testing/images_tissue original_dataset/camelyon16/testing/images_normal
```
4. Randomly select patches on WSIs on level0
```shell
python preprocess/wsi/bin/sampled_spot_gen.py original_dataset/camelyon16/testing/images_tumor/ preprocess/coords/test_bad.txt 400
```
5. Generate patches for training and testing use
```shell
python preprocess/wsi/bin/patch_gen.py original_dataset/camelyon16/training/tumor preprocess/coords/train_bad.txt data/camelyon16/train/bad
```

## Train and test the model
```shell
python main.py original_dataset/camelyon16/testing/images/ checkpoints/0detection_camelyon166.pth preprocess/configs/wideresnet_50.json original_dataset/camelyon16/testing/images_tissue/ result/
```
## Postprocess

```shell
python main.py original_dataset/camelyon16/testing/images/ checkpoints/0detection_camelyon166.pth preprocess/configs/wideresnet_50.json original_dataset/camelyon16/testing/images_tissue/ result/
```
### AUROC Evaluation
### FROC Evaluation
1. Tumor localization
```shell
python preprocess/wsi/bin/nms.py result/modified/ result/csv/
```
2.FROC evaluation
```shell
python preprocess/wsi/bin/Evaluation_FROC.py original_dataset/camelyon16/testing/images_mask result/csv
```
