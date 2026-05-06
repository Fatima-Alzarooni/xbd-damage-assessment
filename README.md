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
### SegFormer:
### Siamese DeepLabV3+:
### Soft Voting Ensemble:

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
