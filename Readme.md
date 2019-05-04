# Efficient pan-cancer whole-slide image classification using convolutional neural networks

Code acompaining paper: [paper name](link)

## Prerequisites

* PyTorch 

## Installing

* Clone this repo to your local machine using:
```
 git clone https://github.com/sedab/PathCNN.git
 
```

## Usage

### 1. Data:

* Download the [GDC data transfer API](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)
* Create a manifest by selecting Cases > CANCER_TYPE and Files > Data Type > Tissue Slide Image.
* Download the manifest into ```manifest_file```
* Run the command ```gdc-client download -m manifest_file``` in Terminal

### 2. Data processing:

Note that data tiling and sorting scripts come from [Nicolas Coudray](https://github.com/ncoudray/DeepPATH/). Please refer to the README within `DeepPATH_code` for the full range of options. Additionally, note that these scripts may take a significant amount of computing power. We recommend submitting sections 2.1 and 2.2 to a high performance computing cluster with multiple CPUs.

#### 2.1. Data tiling
Run ```Tiling/0b_tileLoop_deepzoom2.py``` to tile the .svs images into .jpeg images. To replicate this particular project, select the following specifications:

```sh
python -u Tiling/0b_tileLoop_deepzoom2.py -s 512 -e 0 -j 28 -f jpeg -B 25 -o <OUT_PATH> "<INPUT_PATH>/*/*svs"
```

* `<INPUT_PATH>`: Path to the outer directory of the original svs files

* `<OUT_PATH>`: Path to which the tile files will be saved

* `-s 512`: Tile size of 512x512 pixels

* `-e 0`: Zero overlap in pixels for tiles

* `-j 28`: 28 CPU threads

* `-f jpeg`: jpeg files

* `-B 25`: 25% allowed background within a tile.

#### 2.2. Data sorting
To ensure that the later sections work properly, we recommend running these commands within `<ROOT_PATH>`, the directory in which your images will be stored:

```sh
mkdir <CANCER_TYPE>TilesSorted
cd <CANCER_TYPE>TilesSorted
```

* `<CANCER_TYPE>`: The dataset such as `'Lung'`, `'Breast'`, or `'Kidney'`

Next, run `Tiling/0d_SortTiles.py` to sort the tiles into train, valid and test datasets with the following specifications.

```sh
python -u <FULL_PATH>/Tiling/0d_SortTiles.py --SourceFolder="<INPUT_PATH>" --JsonFile="<JSON_FILE_PATH>" --Magnification=20 --MagDiffAllowed=0 --SortingOption=3 --PercentTest=15 --PercentValid=15 --PatientID=12 --nSplit 0
```

* `<FULL_PATH>`: The full path to the cloned repository

* `<INPUT_PATH>`: Path in which the tile files were saved, should be the same as `<OUT_PATH>` of step 2.1.

* `<JSON_FILE_PATH>`: Path to the JSON file that was downloaded with the .svs tiles

* `--Magnification=20`: Magnification at which the tiles should be considered (20x)

* `--MagDiffAllowed=0`: If the requested magnification does not exist for a given slide, take the nearest existing magnification but only if it is at +/- the amount allowed here (0)

* `--SortingOption=3`: Sort according to type of cancer (types of cancer + Solid Tissue Normal)

* `--PercentValid=15 --PercentTest=15` The percentage of data to be assigned to the validation and test set. In this case, it will result in a 70 / 15 / 15 % train-valid-test split.

* `--PatientID=12` This option makes sure that the tiles corresponding to one patient are either on the test set, valid set or train set, but not divided among these categories.

* `--nSplit=0` If nSplit > 0, it overrides the existing PercentTest and PercentTest options, splitting the data into n even categories. 

#### 2.3. Build tile dictionary

Run `Tiling/BuildTileDictionary.py` to build a dictionary of slides that is used to map each slide to a 2D array of tile paths and the true label. This is used in the `aggregate` function during training and evaluation.

```sh
python -u Tiling/BuildTileDictionary.py --data <CANCER_TYPE> --path <ROOT_PATH>
```
* `<ROOT_PATH>` points to the directory path for which the sorted tiles folder is stored in, same as in 2.2.

Note that this code assumes that the sorted tiles are stored in `<ROOT_PATH><CANCER_TYPE>TilesSorted`. If you do not follow this convention, you may need to modify this code.

### 4. Train model:

In the Load data section of `train.py` (lines ~85-96) please modify these variables:

* `root_dir = "<ROOT_PATH><CANCER_TYPE>TilesSorted/"`, change the path to your file path

* `tile_dict_path = "<ROOT_PATH><CANCER_TYPE>_FileMappingDict.p`, change the path to your tile dict path

**NOTE:** We are using a very useful tool called [CometML](https://www.comet.ml/) to keep track of our experiments. If you would like to use it as well, please modify the API key in the training code to your own API key (line ~57). Otherwise, you can remove the lines of code related to cometml. 

Run `train.py` to train with our CNN architecture. sbatch file `run_job.sh` is provided as an example script for submitting a GPU job for this script.

* `--cuda`: enables cuda

* `--ngpu`: number of GPUs to use (default=1)

* `--data`: data to train on (lung/breast/kidney)

* `--augment`: use data augmentation or not

* `--batchSize`: batch size for data loaders (default=32)

* `--imgSize`: the height / width that the image will be shrunk to (default=299)

* `--metadata`: use metadata or not

**IMPORTANT NOTE: this option is not fully implemented!** Please see section 6 for additional information about using the metadata. 

* `--nc`: input image channels + concatenated info channels if metadata = True (default = 3 for RGB).

* `--niter`: number of epochs to train for (default=25)

* `--lr`: learning rate for the optimizer (default=0.001)

* `--decay_lr`: activate decay learning rate function

* `--optimizer`: Adam, SGD or RMSprop (default=Adam)

* `--beta1`: beta1 for Adam (default=0.5)

* `--earlystop`: use early stopping

* `--init`: initialization method (default=normal, xavier, kaiming)

* `--model`: path to model to continue training from a checkpoint (default='')

* `--experiment`: where to store samples and models (default=None)

* `--nonlinearity`: nonlinearity to use (selu, prelu, leaky, default=relu)

* `--dropout`: probability of dropout in each block (default=0.5)

* `--method`: aggregation prediction method (max, default=average)


### 5. Test model:

Run ```test.py``` to evaluate a specific model on the test data, ```run_test.sh``` is the associated sbatch file.

* `--data`: Data to train on (lung/breast/kidney)
* `--experiment` Name of experiment to test, same as in section 4.1
* `--model`: Name of model to test, e.g. `epoch_10.pth`


## Additional resources:

### iPython Notebooks

* ```100RandomExamples.ipynb``` visualizes of 100 random examples of tiles in the datasets
* ```Final evaluation and viz.ipynb``` provides code for visualizing the output prediction of a model, and also for evaluating a model on the test set on CPU
* ```LungJsonDescription.ipynb``` explores the potential of metadata that can be used as extra information for training
* ```new_transforms_examples.ipynb``` visualizes a few examples of the data augmentation used for training. One can tune the data augmentation here.

