# Vision Transformer (ViT) on CIFAR-10

This repository contains the implementation of a Vision Transformer (ViT) trained on the CIFAR-10 dataset. This project was developed during the MLH Global Hack Week as an introduction to leveraging transformer models for computer vision tasks.

## Features

### Vision Transformer (ViT):

Implementation of a ViT from scratch.

Training and evaluation on the CIFAR-10 dataset.

### Deep Learning Frameworks:

PyTorch used for model creation, training, and evaluation.

## CIFAR-10 Dataset:

Used to benchmark the ViT's performance on image classification tasks.

## Reproducible Results:

Code for loading, training, and testing the model.

Pre-trained model checkpoint included for quick testing.

## Usage

1. **Clone the repository**:
    ```bash
    git clone <repository_url>

3. **Navigate to the project directory**:
   ```bash
   cd ViT-CIFAR10

3. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook ViTransformer.ipynb

4. **For standalone scripts**:

   Train the model: Use vit_model.py.
   
   Load and test a pre-trained model: Use load_vit_model.py.

## Learning Objectives

Understand the architecture and implementation of Vision Transformers.

Learn to preprocess and train on CIFAR-10 using PyTorch.

Experiment with transformer-based models for image classification.

## Results

Accuracy: Achieved 93% accuracy on the CIFAR-10 test set.

Visualization: Includes attention maps and classification results.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
