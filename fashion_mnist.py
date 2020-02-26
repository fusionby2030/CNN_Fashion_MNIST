import keras
#https://keras.io/models/
from keras.models import Sequential,Input,Model
#https://keras.io/layers/core/
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from initial_data_analysis import valid_X, train_X, train_label, valid_label, test_X, test_Y_binary
batch_size = 64 #number of training examples used in one iteration

epochs = 20 # total number of times going through the data
num_classes = 10 #number of labels
"""
After importing the libraries and setting the above variables, we are ready to
start building the archetecture
Since this is for image classification, a convolution layer is utilized
followed by a RELU activation function for non-linear decision boundraries
After the RELU function, we have a pooling class to pool the image data into smaller subsections
After every Pooling layer we have a dropout, so that a fraction of the neurons are randomly turned
off to not allow for memorization of the training data
See the README for a more detailed explanation on the architecture

"""
fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='linear'))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(num_classes, activation='softmax'))


#https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
#compile the model with the loss determined by categorical_crossentropy https://keras.io/losses/
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
#to see the final parameters of the model
fashion_model.summary()
"""
Now we can train the model
"""
fashion_train = fashion_model.fit(train_X, train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))
fashion_model.save("fashion_model.h5py") #saving the model parameters
"""
And evalualte the results
https://keras.io/models/model/
Evaluate will return a list of scalars of the loss and accuracy
"""
test_eval = fashion_model.evaluate(test_X, test_Y_binary, verbose=0)

print(test_eval[0]) #loss
print(test_eval[1]) #accuracy
