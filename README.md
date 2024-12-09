# COMP 432 - Final Project **Group K**
## Image Classification using Convolutional Neural Networks (CNNs) and Transfer Learning

This project explores using Convolutional Neural Networks (CNNs) to address image classification tasks from real-world applications in computational pathology and computer vision. The project is divided into two main tasks:

1. **Training and Analyzing CNN Models on Pathology Data**: 
   - Train CNNs on a Colorectal Cancer Tissue Dataset.
   - Analyze and visualize feature representations using t-SNE.
   - Quantitative analysis of the model's performance
   
2. **Knowledge Transfer and Cross-Dataset Analysis**:
   - Transfer learned feature representations to other datasets (Prostate Cancer and Animal Faces).
   - Compare feature extraction against a pre-trained CNN encoder trained on ImageNet.
   - Use a classical machine learning algorithm (Random Forest) to classify the extracted
     features on the Prostate Cancer and Animal Faces datasets.
   
The main objectives are:
- To study how CNNs generalize across datasets.
- To conduct a detailed analysis of model performance using t-SNE visualizations.
- To classify extracted features using classical machine learning methods.

## Requirements
To set up the environment, use:

```bash
pip install -r requirements.txt
```

## Dataset

The datasets used in this project are:

1. **Colorectal Cancer Tissue Dataset** (used for training CNNs).
2. **Prostate Cancer Dataset** (used for cross-dataset analysis).
3. **Animal Faces Dataset** (used for cross-dataset analysis).

They are already Downloaded if you clone the Github Repository, If you want to download them:

### Download Links
The datasets can be downloaded from the following sources:

- **Colorectal Cancer Tissue Dataset**: [Download Link](https://1drv.ms/u/s!AilzKc-njjP7mN0NOZvxl0TPAUxmig?e=K0TpeX)
- **Prostate Cancer Dataset**: [Download Link](https://1drv.ms/u/s!AilzKc-njjP7mN0M_LjB5xeAydDsrA?e=0obzsx)
- **Animal Faces Dataset**: [Download Link](https://1drv.ms/u/s!AilzKc-njjP7mN0LqoRZvUYONY9sbQ?e=wxWbip) 

### Directory Structure
Once downloaded, organize the datasets into the following directory structure:
   - **Data/Colorectal Cancer**
   - **Data/Prostate Cancer**
   - **Data/Animal Faces**

## Instructions

### 1. Training and Validating Your Model
To train and validate your CNN model on the **Colorectal Cancer Dataset**, follow these steps:

1. Open the `task_1.ipynb` notebook in Jupyter Notebook or JupyterLab.
2. Ensure the **Colorectal Cancer Dataset** is placed in the correct directory: `Data/Colorectal Cancer/`.
3. Execute all the cells in the notebook to:
   - Preprocess the data.
   - Train the CNN model.
   - Validate the model performance on the validation set.
4. Once training is complete, the trained model will be automatically saved as `resnet50_colorectal.pth` in the `task_1/` directory.
5. t-SNE visualizations of the model's feature representations will be saved in the `Figures/` directory.

### 2. Running the Pre-Trained Model
To evaluate the pre-trained model on the **provided test dataset**, follow these steps:

1. Extract the **test dataset** from the project ZIP file.
2. Open the `task_1.ipynb` notebook in Jupyter Notebook or JupyterLab.
3. Locate the section titled **"Evaluate Pre-Trained Model"** in the notebook.
4. Ensure the pre-trained model file, `resnet50_colorectal.pth`, is present in the `task_1/` directory.
5. Modify the `test_dir` variable in the notebook to point to the test dataset directory (`Data/Colorectal Cancer/Test/`).
6. Run the evaluation cells to compute:
   - Performance metrics (e.g., accuracy, precision, recall, F1-score).
   - Visualizations of feature representations on the test dataset.

The evaluation results, including metrics and visualizations, are displayed in the notebook and saved to the `Classification_Reports/` and `Figures/` directories.
