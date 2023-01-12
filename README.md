# FW_RD
## Preprocess
### Annotations
xml annotation to json
```shell
python preprocess/wsi/bin/camelyon16xml2json.py {xml_path} {json_path}
```
### Generate image patches
1. Generate tissue masks for WSIs
```shell
python preprocess/wsi/bin/tissue_mask.py {wsi_path} {tissue_path}
```
2. Generate tumor masks for WSIs
```shell
python preprocess/wsi/bin/tumor_mask.py {wsi_path} {json_path} {tumor_path}
```
3. Generate masks for normal areas
```shell
python preprocess/wsi/bin/non_tumor_mask.py {tumor_path} {tissue_path} {normal_path}
```
4. Randomly select patches on WSIs on level0
```shell
python preprocess/wsi/bin/sampled_spot_gen.py {mask_path} {coords_path} /maximum # of patch for each WSI/
```
You may generate 6 txt files, such as tumor_train.txt, normal_train.txt, tumor_val.txt, normal_val.txt, tumor_test.txt, normal_test.txt

5. Generate patches for training and testing use
```shell
python preprocess/wsi/bin/patch_gen.py {wsi_path} {coords_path} {patch_path}
```

## Train and test the model
Run main.py to train and test the model, put the checkpoints in "checkpoints" file. 

### Anomaly Map Generation
set vis = Ture in main.py
```shell
python main.py {wsi_path} {ckpt_path} {cfg_path} {mask_path} {probs_map_path}
```
cfg file for RD model is under '\preprocess\configs'. {mask_path} indicates the tissue mask. Please put the anomaly maps for tumor and normal WSIs in separate files: {probs_map_path}/good for normal and {probs_map_path}/bad for tumor.

## Postprocess
### AUROC Evaluation
1. AUROC Evaluation
```shell
python preprocess/wsi/bin/AUROC_plot.py {probs_map_path}
```

2.Heatmap Generation
```shell
python preprocess/wsi/bin/AUROC_plot.py {image_path}
```

### FROC Evaluation
1. Tumor localization
```shell
python preprocess/wsi/bin/nms.py {probs_map_path} {coord_path}
```


2.FROC evaluation
```shell
python preprocess/wsi/bin/Evaluation_FROC.py {Camelyon16_test_image_mask} {coord_path}
```
