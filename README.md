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
