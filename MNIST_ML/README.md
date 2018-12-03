# The digit recognition dataset

![](img/images.png)

### Objective
* Build a classifier that predicts the digit from the handwriting.
* Construct model complexity curve: compute and plot the training and testing accuracy scores for a variety of different neighbor values

### Dataset
* 10 classes, the digits 0 through 9.
* A reduced version of the MNIST dataset is one of scikit-learn's included datasets.
* Each sample in this scikit-learn dataset is an 8x8 image representing a handwritten digit. Each pixel is represented by an integer in the range 0 to 16, indicating varying levels of black.
* scikit-learn provides an 'images' key in addition to the 'data' and 'target' keys.
* 'images' key: a 2D array of the images corresponding to each sample.
* 'data' key: contains the feature array - that is, the images as a flattened array of 64 pixels.

## References

* This notebook was inspired by DataCamp course Supervised Learning with scikit-learn [link](https://www.datacamp.com/home)
* The complete dataset is available at MNIST [link](http://yann.lecun.com/exdb/mnist/)