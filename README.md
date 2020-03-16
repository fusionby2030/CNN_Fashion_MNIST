# Simple CNN for Fashion_MNIST data analysis

This is a simple implementation of a Convolution Neural Network(CNN) using Keras/TensorFlow to classify images. The dataset is Fashion-MNIST provided by keras. The main goal of this project is to give an introduction into using CNN's to classify different images, and how to format data to be fed into a Neural Network.

## File Structure

Below is the file structure of the repository.

### Neural Network

The neural network is built in the folder labeled pyimagesearch, which also contains an overview of the structure of the network itself.

### Data Prep

The data is initially inspected in the initial_data_analysis file, however it is ultimately imported directly and split in the fashion_mnist file.
This initial data analysis is important to review, since in my opinion every Neural Network project is 80% data preparation, and 20% Neural Network fun. Therefore, one should fully understand the dataset they are working with before going on to feed it to a neural network.
Ultimately we are trying to classify the following types of images:

![Image of MNIST data](/images/Fashion MNIST_screenshot_03.03.2020.png)

### Training and Prediction

This can be found in the fashion_mnist file.
This file also will output the accuracy and loss of the neural network over time, which will look something like this (varies by machine).

![Plot](/images/plot.png)

### Evaluation

After training through the running of fashion_mnist.py, the model will be saved to a file called fashion_model.h5py which is not included in this repository, but will be needed to run the evaluation file fashion_mnist_loaded.py
This in the end will output the overall accuracy of the model after training, and also the sklearn metrics. To learn more about the metrics output and why they can be usefull to know, see [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support).
