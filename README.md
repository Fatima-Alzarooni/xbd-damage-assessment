# Automated Post-Disaster Building Damage Assessment

This repository contains state-of-the-art semantic segmentation models tailored for post-disaster structural damage assessment applied to satellite imagery. 
Each model is optimized to segment building damage into multiple classes, including no-damage, minor-damage, major-damage, and destroyed, addressing challenges like extreme class imbalance and spatial resolution loss. 
The different standalone baselines are explored to compare their individual performance against a unified weighted ensemble approach, evaluating how they perform in time-sensitive emergency response scenarios.

## All the models have Common Features such as:

* **Input:** Pre- and post-disaster satellite image pairs from the xBD dataset.
* **Output:** Multi-class semantic segmentation heatmap (4 damage levels).
* **Imbalance Handling:** Handled via architectural strengths, specialized loss functions, and a multi-model fusion mechanism.
* **Evaluation:** Monitored using Pixel Accuracy, Mean Intersection over Union (mIoU), F1-Score, and comprehensive confusion matrices.

## All of the codes are structured in the following Way:

* **SECTION 1: Data Preparation & EDA:** (`download.py`, `dataset.py`, `preprocessing.ipynb`, `eda.ipynb`)
* **SECTION 2: Data Augmentation:** (`augment.py`)
* **SECTION 3: Model Architectures:** (`unet_model.py`, `my_segformer.py`, `deepLabV3Plus.py`, `z-segformer.py`)
* **SECTION 4: Training Loops & Logs:** (`train_unet_base.py`, `train_segformer.py`, `train_deepLabV3Plus.py`, and `.txt`/`.json` logs)
* **SECTION 5: Testing & Evaluation:** (`test_siamese_unet.ipynb`, `test_segformer.ipynb`, `test_deeplabv3.ipynb`)
* **SECTION 6: Ensemble Implementation:** (`damage_ensemble.py`, `test_ensemble.ipynb`)
* **SECTION 7: Visualizations:** Located in `checkpoints/` directories (e.g., `miou.png`, `f1.png`, `confusion_matrix_test.png`, `sample_predictions.png`)

## The different models we used are:

### Siamese U-Net:
The **Siamese U-Net** is a specialized Encoder-Decoder architecture designed to process both pre-disaster and post-disaster images simultaneously to extract temporal structural changes. 
It leverages a dual-stream contracting path (encoder) to extract features, followed by an expanding path (decoder) that progressively upsamples these features. 
A critical component is the use of skip connections, which fuse high-resolution spatial features from the encoder with the upsampled decoder features. 
This allows the model to localize objects, preserve fine-grained spatial details, and recover sharper, distinct building boundaries that are crucial for distinguishing individual structures in dense residential areas.

### SegFormer:
The **SegFormer** architecture adapts transformer-based deep learning for semantic segmentation. 
By discarding traditional positional encodings and utilizing a hierarchically structured transformer encoder paired with a lightweight Multi-Layer Perceptron (MLP) decoder, it efficiently aggregates multi-scale features. 
This mechanism allows the model to capture wide, global context across the disaster zones, making it robust to complex terrain changes and overarching disaster patterns across the satellite imagery.

### Siamese DeepLabV3+:
The **Siamese DeepLabV3+** is a semantic segmentation architecture tailored for macro-level contextual awareness. 
It enhances standard segmentation by integrating an ASPP (Atrous Spatial Pyramid Pooling) module, which uses dilated convolutions to gather multi-scale context. 
This allows the model to localize wide-scale zones of catastrophic damage without hallucinating buildings in empty terrain. 
While it occasionally sacrifices fine spatial resolution (resulting in overlapping building footprints), it is a powerful feature extractor for identifying severe structural damage.

### Soft Voting Ensemble:
The **Soft Voting Ensemble** pipeline is the culmination of the project, combining the three trained models (Siamese U-Net, Siamese DeepLabV3+, and SegFormer). 
Implemented via `damage_ensemble.py`, this approach operates at inference time. 
Each model produces a set of class logits for a given pre/post image pair, which are converted to probability distributions via a softmax function. 
These three probability maps are then averaged element-wise, and the final damage class for each pixel is determined by taking the argmax of the averaged probabilities. 
By aggregating predictions at the probability level rather than the decision level, this soft voting method is significantly more robust to individual model uncertainty than hard majority voting, particularly for ambiguous intermediate damage classes.

## How to Run the Code (Step-by-Step)

### Step 1: Environment Setup
Ensure you have Python installed, along with standard deep learning libraries (PyTorch, Torchvision, Transformers, etc.). If a requirements file is present, install the dependencies using:
`pip install -r requirements.txt`

### Step 2: Data Preparation
The xBD dataset must be downloaded and preprocessed before training.
1. Run `python download.py` to fetch the required dataset files.
2. Open and execute all cells in `preprocessing.ipynb`. This will pair the pre- and post-disaster images, map the 4-level damage labels, and format the ground-truth masks for training.

### Step 3: Training the Models
You can train each baseline model individually. The Python scripts are configured to automatically log training metrics and save the optimal model weights (`.pth`) based on validation performance. Run these from your terminal:
* **To train Siamese U-Net:** `python train_unet_base.py`
* **To train SegFormer:** `python train_segformer.py`
* **To train Siamese DeepLabV3+:** `python train_deepLabV3Plus.py`

### Step 4: Testing & Evaluation
Testing and metric calculation (mIoU, F1-Score, Confusion Matrices) are handled in Jupyter Notebooks to allow for immediate visual analysis of the segmentation maps.
1. Open `test_siamese_unet.ipynb`, `test_segformer.ipynb`, or `test_deeplabv3.ipynb` in your preferred environment (e.g., VS Code or Jupyter).
2. Ensure the weight paths in the notebook point to your newly trained `.pth` files.
3. Run all cells to evaluate the test set and generate prediction heatmaps.

### Step 5: Running the Ensemble
Once all three standalone models are fully trained and their `.pth` weights are saved:
1. Open `test_ensemble.ipynb` (or execute `damage_ensemble.py`).
2. Verify that the file paths correctly point to the weights for all three individual models.
3. Run the code. The script will load all three architectures, compute the element-wise average of their softmax probability maps, and output the final comparative metrics for the unified ensemble pipeline.
