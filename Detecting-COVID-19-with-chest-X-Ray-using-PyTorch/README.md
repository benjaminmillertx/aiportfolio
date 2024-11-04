COVID-19 Detection via Chest X-Ray Imaging
A deep learning model for multi-class classification of chest X-ray scans, distinguishing between Normal, Viral Pneumonia, and COVID-19 cases.



Project Overview
This Coursera project focuses on developing a robust model that categorizes chest X-ray images into three distinct classes: Normal, Viral Pneumonia, and COVID-19. The dataset, sourced from Kaggle, contains 2,924 grayscale images, with a significant class imbalance, particularly in the COVID-19 category. The dataset details are as follows:

1,341 Normal images
1,345 Viral Pneumonia images
219 COVID-19 images
Link to dataset on Kaggle

Methodology
Data Splitting: The dataset was partitioned to include a validation subset of 90 images (30 per class) to evaluate the model’s accuracy during training.

Model Architecture:
Leveraging a pre-trained ResNet18, the final classification layer was adjusted to predict among the three classes. Inspired by prior projects, like Skin Cancer Classification, the entire model was fine-tuned during training for optimal performance.

Training Configuration:

Optimizer: Adam with a learning rate of 3e-5
Loss Function: PyTorch Cross-Entropy
Batch Size: 6
Image Preprocessing: Resized to (224, 224) and normalized as per model specifications
The model exhibited rapid convergence, achieving over 98% validation accuracy within a single epoch.



Using the Notebook
Download the dataset from Kaggle.
Download and execute the notebook.
Results
The model achieved an average accuracy of 98%, including a perfect 100% accuracy rate in detecting COVID-19 cases.