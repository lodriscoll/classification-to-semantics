# From Classification to Semantics with CNNs

This repository contains a Jupyter notebook that demonstrates the transition from image classification to semantic segmentation using Convolutional Neural Networks (CNNs). The notebook was developed by Liam O'Driscoll under the direction of Assistant Professor Cihang Xie at UCSC.

## Introduction
The notebook provides a comprehensive review of image classification techniques and then dives into the implementation of semantic segmentation. It explores various CNN architectures and their applications in computer vision tasks.

## Setup
Before running the notebook, make sure you have the necessary packages installed. The notebook utilizes TensorFlow and Keras for implementing the models. Additional packages such as matplotlib, numpy, and PIL are also used for data visualization and processing. This notebook is contained and can be run entirely in a google colab environment, there is no need to download data or preprocess it on your local runtime. 

## Data
For the classification part of the exercise, the CIFAR-10 dataset is used. This dataset consists of 60,000 32x32 color images in 10 different classes. The notebook loads the dataset, preprocesses the data, and performs necessary transformations for training and evaluation.

## Classification
The notebook starts by building a fully connected neural network (MLP) for image classification. It trains the MLP model on the CIFAR-10 dataset and evaluates its performance. The limitations of MLPs for image classification are discussed, and potential improvements are suggested.

## Convolutional Neural Networks (CNNs)
To overcome the limitations of MLPs, the notebook introduces the concept of CNNs. It explains the benefits of convolutional operations, such as parameter sharing and translation invariance, and their impact on improving efficiency and location-independent pattern recognition.

The notebook then constructs a CNN model with multiple convolutional and max-pooling layers, followed by a better MLP classifier. The CNN model is trained and evaluated on the CIFAR-10 dataset, demonstrating its superior performance compared to MLPs.

## Residual Networks (ResNet)
To further improve the performance of CNNs, the notebook introduces the concept of residual layers and the ResNet50 architecture. ResNet50 is a deep CNN architecture that leverages residual learning to capture both low-level and high-level features in images.

The notebook explains how ResNet50 can be used as a pre-trained model, fine-tuned, and transferred to specific datasets like CIFAR-10. It provides the necessary code to build a transfer learning model using ResNet50 and a better MLP classifier for classifying the CIFAR-10 dataset.

## Semantic Segmentation
Using the techniques discussed in the classification section of this notebook, it then implements semantic segmentation using the cityscapes dataset. A transfer learning approach is first used to get coarse semantic masks. Transpose convolution is then introduced to get fine grain semantic masks. Finally, the U-NET architecture is introduced to improve on performance. 

## Conclusion
This notebook serves as a comprehensive guide to understanding the transition from image classification to semantic segmentation using CNNs. It provides insights into the limitations of traditional MLPs, the advantages of CNNs, and the effectiveness of transfer learning with architectures like ResNet50.

Feel free to explore the notebook, experiment with different models and techniques, and gain a deeper understanding of the world of computer vision with CNNs.

