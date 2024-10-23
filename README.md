# COMP 432 - Project Proposal **Group K**

## 1. Problem Statement and Application

In machine learning, transferability refers to the ability of a model trained on one dataset to generalize and perform well on different datasets. This is crucial because, in many domains, large labelled datasets are not always available for every specific task. Instead of retraining a new model for each dataset from scratch, transferring knowledge learned from one dataset can save time, resources, and computational power.

This project investigates these challenges by exploring the transferability of convolutional neural networks (CNNs) across different datasets. The study focuses on how well a model trained on colorectal cancer images can generalize to prostate cancer and a non-medical dataset. The project will employ and modify an existing CNN architecture to suit different tasks and evaluate its performance using quantitative and qualitative metrics.

## 2. Reading Material

To provide context and background for this project, the following reading materials will be examined:

- **Paper 1:** This paper discusses the use of deep learning techniques, specifically the ResNet-18 and ResNet-50 architectures, for image classification in the detection of colorectal cancer.[3]
- **Paper 2:** This study evaluates the ResNet-50 model for prostate cancer detection, achieving high accuracy ratings. [6]
- **Paper 3:** The study shows that domain adaptive machine learning models can effectively transfer and generalize maize yield predictions across different regions and time periods. [4]

## 3. Possible Methodology

### 3.1 Task 1: Colorectal Cancer Classification

We will modify a CNN model (like ResNet-34) to categorize images into three groups: smooth muscle, normal colon tissue, and cancer-related stroma. To improve performance and avoid overfitting due to the small dataset, we will use techniques like random rotations, flips, and color changes to the images. We’ll also add methods like weight decay and dropout to make the model more stable and better at generalizing to new data. A learning rate scheduler will help optimize the training process.

### 3.2 Task 2: Feature Extraction and Transferability

Next, we’ll use the trained CNN model from Task 1 to analyze two new datasets without additional training. We will examine the features of the new data using a tool called t-SNE and compare the results with a model pre-trained on ImageNet. We’ll visualize how the features are distributed and then use a machine learning classifier to categorize features for prostate cancer and animal faces, selecting the best model for each dataset.

## 4. Metric Evaluation

### 4.1 Quantitative Analysis

For the quantitative evaluation, we will measure how well the model classifies images using accuracy—the percentage of correct classifications. We expect better accuracy with the colorectal cancer data and slightly lower accuracy on transfer learning tasks. Additionally, precision and F1-score will be employed to assess the model’s performance, particularly in the medical domain, where it is critical to minimize both false positives and negatives.

### 4.2 Qualitative Analysis

For the qualitative analysis, we will apply t-SNE visualizations to examine the feature distribution across different classes, providing insights into how well the model has learned to separate them. This method will be especially beneficial for evaluating transfer learning performance. In parallel, we will plot learning curves for training and validation accuracy and loss, allowing us to monitor convergence during training and identify any potential overfitting or underfitting issues.

## Gantt Chart

| Name | Begin Date | End Date |
| --- | --- | --- |
| GitHub Setup | 2024-09-27 | 2024-09-27 |
| Revise Proposal | 2024-09-27 | 2024-10-06 |
| Submit Final Proposal | 2024-10-07 | 2024-10-07 |
| Milestone 1 | 2024-10-07 | 2024-10-07 |
| Train Model on Colorectal Cancer Dataset | 2024-10-07 | 2024-10-12 |
| Analyze and Visualize Results on Phase 1 Training | 2024-10-13 | 2024-10-19 |
| Milestone 2 | 2024-10-20 | 2024-10-20 |
| Submit Progress Report | 2024-11-03 | 2024-11-03 |
| Apply and Visualize Results of CNN on Set 2 & 3 | 2024-10-20 | 2024-10-27 |
| Apply ImageNet CNN on Dataset 2 & 3 | 2024-10-28 | 2024-11-04 |
| Analyze CNN vs. ImageNet Result | 2024-11-05 | 2024-11-12 |
| Apply Classical ML Technique to Classify Dataset 2 & 3 | 2024-11-13 | 2024-11-20 |
| Submit Final Report | 2024-11-21 | 2024-12-01 |
| Submit Final Presentation | 2024-11-21 | 2024-12-01 |
| Milestone 3 | 2024-12-01 | 2024-12-01 |

## References

1. Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, and Mu Li. Bag of tricks for image classification with convolutional neural networks. arXiv preprint arXiv:1904.10423, 2019.
2. M. A. Khan, A. Hussain, A. Majid, M. Yasmin, Z. Mehmood, and R. Abbasi. Deep learning in image classification using residual network (resnet) variants for detection of colorectal cancer. Journal of Medical Systems, 43:225, 2019.
3. Y. Liu, D. Zhang, Z. Wang, and Y. Li. A deep learning method for breast cancer detection using histopathological images. Procedia Computer Science, 175:652–658, 2021.
4. Rhorom Priyatikanto, Yang Lu, Jadu Dash, and Justin Sheffield. Improving generalisability and transferability of machine-learning-based maize yield prediction model through domain adaptation. Agricultural and Forest Meteorology, 341:109652, 2023.
5. M. M. Rahman, Y. Wang, and B. Zheng. Prostate image classification using pretrained models: Googlenet and resnet-50. International Journal of Computer Science and Network Security (IJCSNS), 19(10):88–93, 2019.
6. K. Sivakumar, R. Jothi, N. Elangovan, and M. Natarajan. A hybrid deep learning model for the detection of colon cancer. IEEE Access, 10:13351–13363, 2021.
