# boxplot
the Goal is that we can detect changes of underlying scanner/distribution and use this information to stabilize the training to avoid catastrophic forgetting\
in the [plot] to plot samples and their mean/std dev.\
in the [mode_tran] is the model about how to use U-Net model to train the dataset, and then show the change about the data from different scanners.
#Anomaly Detection Method
1. Overview
We employ Gaussian Discriminant Analysis (GDA) to model the feature distribution of deep models and use Mahalanobis Distance for confidence estimation. Compared to the traditional Softmax confidence method, this approach provides a more effective way to distinguish in-distribution (ID) and out-of-distribution (OOD) samples.

2. Methodology
a. Constructing Class-Conditional Gaussian Distributions
The features extracted from the deep model for each class are assumed to follow a Gaussian distribution.
This distribution is characterized by the mean and covariance of the feature vectors within each class.


b. Computing Mahalanobis Distance for Confidence Estimation
For a test sample, we measure how far its feature representation deviates from each class’s distribution using the Mahalanobis distance.
The confidence score is defined based on this distance, where a smaller distance indicates higher confidence in the sample belonging to a known class


c. OOD Sample Detection
Confidence scores derived from posterior distributions help detect OOD samples.
Unlike Softmax-based confidence, this approach provides a more stable and reliable distinction between normal and anomalous samples.
