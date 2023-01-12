# SLFCD
## Preprocess
###Annotations
xml annotation to json
```shell
python preprocess/wsi/bin/camelyon16xml2json.py {original_dataset/camelyon16/testing/lesion_annotations/} {preprocess/jsons/test/}
```
###Generate image patches
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

