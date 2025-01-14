# SAR Ship Detection with Complex-Valued Neural Networks

 

This repository has been developed as part of the Master thesis project *SAR Ship Detection with Complex-Valued Neural Networks* conducted at the Netherlands Aerospace Centre (NLR) and University of Amsterdam (UvA). It investigates the effectiveness of complex-valued deep learning and, in a broader sense, the exploitation of phase information for ship detection models that train on Synthetic Aperture Radar (SAR) data. 

 

## How to use this repository

 

### 1. Create conda environment

Create a conda environment (cvnn4sar) with the necessary packages.

 

```bash

bash ./setup/setup.sh

```

 

### 2. Change paths in run.sh

Change the necessary paths in the cvnn/run.sh script. 



2.1. Change the Python path (`PYTHON`) to the path of the Python instance in the cvnn4sar environment.

2.2. Change the pipeline path (`PIPELINE_PATH`) to match the path to ./cvnn/pipeline.

2.3. (Optional) If necessary, change the data root directory (`DATA_ROOT_DIR`) to match the location of the raw (SARFish and Sentinel) products.

 

### 3. run.sh

Run (parts of) the detection pipeline with the run.sh script.

 

```bash

bash ./cvnn/run.sh

```

The pipeline consists of three phases (*pre-processing*, *training*, *testing*) that can be included or excluded by setting the corresponding flags (`do_detect_pre`, `do_detect_train`, `do_detect_test`) to `true` or `false`. Note that each phase requires at least one prior run of the previous phase. Below each phase is explained in more detail.

 

**3.1 pre-processing**

`1_preprocess.py` creates a Pytorch dataset of tiles from the raw full-size Sentinel products. The dataset consists of a folder with all tiles  and for each fold a .json file with for each tile a split-flag (*train*, *validation*, *test*) and its detection targets (*bounding box*, *class label*, *confidence*). If the pre-processing scripts gets interrupted or crashes, a subsequent run will take of from the point it got stuck.

 

Flags:

- `data_root_dir` Root directory of the raw SARFish and Sentinel products. These should be located on the mounted sasa028 drive.

- `data_quality` Which data quality to use. Either the uncalibrated `SARFish_data` (only debursted) or the calibrated `SNAP_data` (radiometric calibrated, debursted, deramped and demodulated).\

- `product_type` Sentinel-1 product type. only `SLC` is implemented at the moment, but code allows for extension to `GRD`.

- `fold` Selection and split (train, validation ,test) of the products. Predefined choices are `fold` (all 50 products, 5 different 80/10/10 splits) and - `smallfold` (selection of 10 products, 5 different 80/10/10 splits).

- `fold_file` The .csv file corresponding to the chosen fold.

- `fold_numbers` Which splits to use from the fold file. For the predefined folds choose from `1-5`,

multiple choices possible.\

- `tile_path` Path to store the created tiles and .json files.

- `tile_size` The size of the constructed tiles. Default is 128x128 pixels.

- `tile_overlap` Overlap of the tiles. Default is 16 pixels.

- `num_augmentations` The number of phase-rotation augmentations to apply to the data. With $N=1$ only the original phase $\varphi_0$ is used. For $N>1$ a copy of each tile with a phase $\varphi_n = \varphi_0 + 2\pi\cdot n/N$, $n \in \{1, ... , N\}$ is made.

 

**3.2 training**

`2_train.py` trains the detection model(s) on the dataset. The model with the lowest loss on the validation set is saved.

 

Flags:

- `dataset_file` Path to the .json file corresponding to the fold that is used for training.

- `representation` Representation of the data that the model trains on. Five representations are possible: `amp_only` (only uses the amplitude of the data), `phase_only` (only uses the phase of the data), `real_imag` (splits the real and imaginary part of the data over channels), `amp_phase` (splits the amplitude and phase over channels), `complex` (uses the complex data as is; the model itself is also complex-valued).

- `drop_low` whether to drop low confidence detections. `1` for yes, `0` for no.

- `drop_empty` whether to drop tiles with no detections. `1` for yes, `0` for no. Has to be `1` in the current set up

- `trained_model_path` Path to the saved model. Pretrained models used in the thesis are provided.

- `tensorboard_folder` Path to save training and validation loss details.

- `frozen_backbone` path to a pretrained backbone model, that gets frozen during training. `None` if backbone should not be frozen (recommended).

- `num_epochs` Number of training epochs. Default is 10.

- `batch_size` Size of training batch. Default is 8.

- `save` whether to save (and potentially overwrite) the trained model.



**3.3 testing**

`3_test.py` evaluates the model on the test data. The detection precision, recall and F_1 scores are computed as well as a confusion matrix for the classification. Identical flags should be kept the same between `2_train.py` and `3_test.py`

 

Flags:

- `dataset_file` See 3.2.

- `representation` See 3.2.

- `drop_low` See 3.2.

- `drop_empty` See 3.2.

- `trained_model_path` See 3.2.

- `tensorboard_folder` The test results will also be stored in this folder.

- `model_to_use` Path to the pretrained model to evaluate.

- `score_thresholds` Detection score thresholds at which to compute the precision and recall.

- `iou_threshold` Intersection over Union (IoU) threshold for considering a detection a true positive.

- `save` whether to save  (and potentially overwrite) the tests results.

 

### 4. Performance demo

To get a feeling for the detection performance of the model a `performance_demo.ipynb` is provided in the pipeline directory. Here the ground truth and model predictions are visualised.

 

## Contact

For inquiries, suggestions or bug reports, please don't hesitate to contact the author.

Job de Witte (job.de.witte@zeelandnet.nl)