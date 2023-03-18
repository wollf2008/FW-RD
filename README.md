# FW-RD
## Prerequisites
### Environment
* Python (3.8)
* Numpy (1.20.3)
* Scipy (1.7.1)
* [PyTorch (0.3.1)/CUDA 8.0](https://pytorch.org/previous-versions/)
* torchvision (0.2.0)
* PIL (8.3.2)
* scikit-image (0.13.1)
* [OpenSlide 3.4.1](https://openslide.org/)
* matplotlib (2.2.2)
* sklearn (1.0)

### Dataset
The main data are the whole slide images (WSI) in `*.tif` format from the [Camelyon16](https://camelyon17.grand-challenge.org/) challenge. You may also download the dataset at [GigaDB](http://gigadb.org/dataset/100439). There are 400 WSis in total, together about 700GB+. Once you download all the slides, please put all the tumor slides and normal slides for training under one same directory.

The Camelyon16 dataset also provides pixel level annotations of tumor regions for each tumor slide in xml format. You can use them to generate tumor masks for each WSI.

# Model
![FW-RD](/image/1.2.png)
This is the main structure of our project. In order to train and test our model, we need to crop WSIs in to 256*256 small patches. The selection and cropping process are shown in Preprocess part.

## Preprocess
In order to train WSIs with deep learning models, we need to crop WSIs into 256*256 patches.

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
I have included the coordinates of pre-sampled patches used in the paper for training and testing. They are located at [perprocess/coords](perprocess/coords).

5. Generate patches for training and testing use
```shell
python preprocess/wsi/bin/patch_gen.py {wsi_path} {coords_path} {patch_path}
```
Please save the generated patches with following sequence:
```
├── data/
│   ├── camelyon16/
│   │   ├── train/
│   │   │   ├── good/
│   │   │   ├── bad/
│   │   ├── test/   
│   │   │   ├── good/
│   │   │   ├── tumor/
│   │   ├── val/   
│   │   │   ├── good/
│   │   │   ├── tumor/
```

## Train and test the model
Run main.py to train and test the model, put the checkpoints in "checkpoints" file. 
```shell
python main.py
```

### Anomaly Map Generation
set vis = Ture in main.py
```shell
python main.py {wsi_path} {ckpt_path} {cfg_path} {mask_path} {probs_map_path}
```
cfg file for RD model is under '\preprocess\configs'. {mask_path} indicates the tissue mask for test WSIs, you can generate them follow the instruction in [Camelyon17](https://camelyon17.grand-challenge.org/) challenge. Please save the anomaly maps for tumor and normal WSIs with following path:
```
├── results/
│   ├── good/
│   ├── tumor/
```

## Postprocess
Note, Test_049 and Test_114 are excluded from the evaluation as noted by the Camelyon16 organizers.

### AUROC Evaluation
1. AUROC Evaluation
```shell
python preprocess/wsi/bin/AUROC_plot.py {probs_map_path}
```

2.Heatmap Generation
```shell
python preprocess/wsi/bin/heatmap_plot.py {prob_map}
```
![FW-RD](/image/heatmap.PNG)

### FROC Evaluation
1. Tumor localization
We use non-maximal suppression (nms) algorithm to obtain the coordinates of each detectd tumor region at level 0 given a probability map.
```shell
python preprocess/wsi/bin/nms.py {probs_map_path} {coord_path}
```
Where {probs_map_path} is where you saved the generated probability map, and {coord_path} is where you want to save the generated coordinates of each tumor regions at level 0 in csv format. There is an optional command --level with default value 6, and make sure it's consistent with the level used for the corresponding tissue mask and probability map.

2.FROC evaluation
With the coordinates of tumor regions for each test WSI, we can finally evaluate the average FROC score of tumor localization.
```shell
python preprocess/wsi/bin/Evaluation_FROC.py {Camelyon16_test_image_mask} {coord_path}
```
{Camelyon16_test_image_mask} is where you put the ground truth tif mask files of the test set, and {coord_path} is where you saved the generated tumor coordinates. Evaluation_FROC.py is based on the evaluation code provided by the Camelyon16 organizers with minor modification. 

 ## Reference
	@InProceedings{MIDL2023,
    author    = {He, Yinsheng and Li, Xingyu},
    title     = {Whole-slide-imaging Cancer Metastases Detection and Localization with Limited Tumorous Data}


