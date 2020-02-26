#Keras supplies many datasets in their library aptly named datasets
from keras.datasets import fashion_mnist #https://keras.io/datasets/

#We store the training and testing images in the following variables

(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()


import numpy as np #for linear algebra



#To see what dimensions I am working with:
print(train_X.shape, train_Y.shape)
#(60000, 28, 28), (60000,)
print(test_X.shape, test_Y.shape)
#(10000, 28, 28), (10000,)
"""
So training data is 60000 samples with each sample havinga 28x28 pixel dimension
and the test data is 10000 samples with the same dimensions

Next we find the number of unique labels in the output using numpy
"""
print(np.unique(train_Y), len(np.unique(train_Y)))
#(array([0,1,2,3,4,5,6,7,8,9]), 10)
"""
So there are 10 total output labels ranging from 0 - 9
Now we want to reshape the images into a matrix of 28x28x1 to feed the CNN
and then normalize the data into a float format ranging from 0-1.
"""
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)
#Reshapes to ((60000, 28, 28, 1), (10000, 28, 28, 1))

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

train_X = train_X / 255.0
test_X = test_X / 255.0

"""
Now the NN is not going to understand the labels in the train_Y and test_Y
the way they are now, so we must transform the labels into a 'vector'
i.e if the output label is 1, we want the vector to be [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
and if the label is 9 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
"""
from keras.utils import to_categorical #https://keras.io/utils/
#converts an array into a binary class matrix
train_Y_binary = to_categorical(train_Y)
#Converts all to the binary representation
test_Y_binary = to_categorical(test_Y)
#both have size (N, 10) where N is number of samples - 60000 for train and 10000 for test
"""
Always important in ML is splitting of the data into two parts, one for training and one for validation
I choose a 80% to 20% split of training data to validate data respectively
In addition, we can randomize the data order with sklearns train_test_split

"""
from sklearn.model_selection import train_test_split  #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_binary, test_size=0.2, random_state=13)
#train_X.shape: (48000, 28, 28, 1), valid_X.shape: (12000, 28, 28, 1), train_label.shape: (48000, 10), valid_label.shape: (12000, 10)
#Now we are ready to use the data:
