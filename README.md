Modeling Human Annotator Uncertainty in Image Classification using CNNs
Overview

This project investigates how deep learning models can learn and reproduce human uncertainty in image classification tasks instead of predicting only a single hard label. Traditional image classification systems are trained using one-hot labels, where every image belongs to exactly one class. However, real-world human annotation behavior often contains ambiguity, disagreement, and uncertainty.

To address this problem, this project uses the CIFAR-10H dataset, which extends the CIFAR-10 benchmark by providing probabilistic labels collected from multiple human annotators. The goal is to train convolutional neural networks that can predict full human label distributions rather than only majority-vote class labels.

The project explores:

Human disagreement patterns in image classification
Entropy-based uncertainty analysis
Soft-label learning
Comparison of multiple loss functions
Comparison between a custom CNN and ResNet-20
Distribution-level evaluation metrics

The work focuses on understanding whether neural networks can model uncertainty in a way that aligns with human perception.

Objectives

The primary objectives of this project are:

Analyze uncertainty and disagreement among human annotators.
Train deep learning models on probabilistic labels.
Compare the effectiveness of different loss functions.
Evaluate how well models capture uncertainty distributions.
Study the relationship between image ambiguity and model predictions.
Dataset
CIFAR-10

The project uses the CIFAR-10 dataset containing:

60,000 color images
10 image classes
Image size: 32×32

Classes:

Airplane
Automobile
Bird
Cat
Deer
Dog
Frog
Horse
Ship
Truck
CIFAR-10H

The CIFAR-10H dataset provides human soft labels for CIFAR-10 test images. Instead of assigning a single class to each image, the dataset contains probability distributions representing how groups of humans classified each image.

Example:

[0.70, 0.20, 0.05, 0.05, ...]

This means:

70% of annotators selected one class
20% selected another
Remaining annotators disagreed further

These distributions allow the model to learn ambiguity and uncertainty.

Human Uncertainty Analysis

The project computes entropy for every image to measure annotator disagreement.

Low entropy:

Humans strongly agree on the class
Easier images

High entropy:

Humans disagree significantly
Ambiguous or difficult images

The notebook includes:

Entropy distribution visualization
Per-class entropy analysis
Confusion-style probability matrix
Low vs high disagreement image comparisons
Model Architectures
Custom CNN

A custom convolutional neural network was implemented with:

Multiple convolution layers
Max pooling layers
Dense layers
Dropout regularization

Architecture summary:

Conv2D → ReLU → MaxPooling
Conv2D → ReLU → MaxPooling
Conv2D → ReLU → MaxPooling
Flatten
Dense
Dropout
Output Softmax Layer
ResNet-20

The project also implements a ResNet-20 inspired architecture using residual blocks.

Key features:

Residual skip connections
Batch normalization
ReLU activation
Deeper architecture compared to the custom CNN

Residual learning helps improve gradient flow and enables better feature extraction.

Training Strategy
Phase 1 — Pretraining

Both models are first pretrained using hard labels from CIFAR-10 with categorical cross-entropy.

Purpose:

Learn general image representations
Stabilize later soft-label training
Phase 2 — Soft Label Fine-Tuning

Models are then fine-tuned using human probabilistic labels.

The project compares three training objectives:

1. KL Divergence Loss

Measures divergence between predicted and true probability distributions.

2. Cross Entropy Loss

Treats soft labels as target probability distributions.

3. Custom Entropy-Aware Loss

A custom loss combining:

KL divergence
Entropy matching penalty

This encourages models not only to predict the correct distribution but also to match the uncertainty level present in human annotations.

Data Augmentation

To improve generalization, the following augmentations are used:

Horizontal flipping
Width shifting
Height shifting
Regularization and Optimization

The project uses:

Adam optimizer
Early stopping
Model checkpointing
Dropout
Batch normalization

These techniques help prevent overfitting and stabilize training.

Evaluation Metrics

The project evaluates models using both distribution similarity and uncertainty alignment metrics.

Distribution Metrics
KL Divergence

Measures divergence between predicted and target distributions.

Lower values are better.

Jensen-Shannon Divergence (JSD)

Measures similarity between probability distributions.

Lower values indicate better alignment.

Cosine Similarity

Measures angular similarity between distributions.

Higher values are better.

Entropy Metrics
Pearson Correlation

Measures linear correlation between:

Human entropy
Predicted entropy
Spearman Correlation

Measures rank correlation between entropy values.

Precision@K

Evaluates whether the model correctly identifies the most uncertain images.

Examples:

Precision@100
Precision@200
Precision@500
Visualizations

The notebook includes several visual analyses:

Entropy histograms
Class-wise uncertainty plots
Confusion-style probability matrices
Training and validation loss curves
KL divergence comparison plots
True vs predicted entropy scatter plots
Uncertainty ranking visualizations

These visualizations help interpret how models learn human uncertainty.

Key Insights

Some important observations from this project include:

Certain image categories naturally produce higher human disagreement.
High-entropy images are visually ambiguous and difficult to classify.
Models trained with soft labels capture richer information than hard-label training.
KL divergence performs better for distribution matching tasks.
Entropy-aware losses help align model uncertainty with human uncertainty.
ResNet-20 generally provides stronger representation learning compared to the custom CNN.
Technologies Used
Programming Language
Python
Libraries and Frameworks
TensorFlow
Keras
NumPy
Pandas
Matplotlib
Seaborn
SciPy
Scikit-learn
Installation

Clone the repository:

git clone https://github.com/your-username/your-repository.git

Move into the project directory:

cd your-repository

Install dependencies:

pip install -r requirements.txt
Running the Project

Launch Jupyter Notebook or Google Colab and run:

human_annotator_project.ipynb

The notebook will:

Download CIFAR-10H labels
Load CIFAR-10 data
Train models
Evaluate uncertainty alignment
Generate visualizations
Future Improvements

Potential future extensions include:

Vision Transformers for uncertainty modeling
Bayesian neural networks
Monte Carlo dropout uncertainty estimation
Calibration analysis
Active learning using uncertainty scores
Larger human annotation datasets
Ensemble uncertainty estimation
Research Relevance

This project connects deep learning with human-centered AI research.

Applications include:

Medical image diagnosis
Autonomous driving
Human-AI collaboration
Ambiguous image understanding
Decision support systems
Trustworthy AI systems

Understanding uncertainty is critical in safety-sensitive environments where overconfident predictions can be harmful.

Conclusion

This project demonstrates that deep learning systems can move beyond hard-label classification and learn richer probabilistic representations aligned with human perception.

By incorporating human uncertainty into training objectives, models become better at:

Representing ambiguity
Modeling disagreement
Predicting calibrated uncertainty
Understanding difficult samples

The comparison between architectures and loss functions provides insight into how neural networks can better approximate human decision-making behavior.

Author

Chervith Nannuru

License

This project is intended for educational and research purposes.
