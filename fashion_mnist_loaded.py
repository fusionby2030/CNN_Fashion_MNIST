from numpy import loadtxt
from keras.models import load_model
from keras.datasets import fashion_mnist
from keras import backend as K
from keras.utils import np_utils
from sklearn.metrics import classification_report
# load model
model = load_model('model.h5')
#Will print a model Summary
model.summary()
# load dataset like in fashion_mnist.py
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
if K.image_data_format() == "channels_first":
	trainX = trainX.reshape((trainX.shape[0], 1, 28, 28))
	testX = testX.reshape((testX.shape[0], 1, 28, 28))
else:
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0
trainY = np_utils.to_categorical(trainY, 10)
testY = np_utils.to_categorical(testY, 10)


# evaluate the model
score = model.evaluate(testX, testY, verbose=0)


print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
preds = model.predict(testX)


labelNames = ["top", "trouser", "pullover", "dress", "coat",
	"sandal", "shirt", "sneaker", "bag", "ankle boot"]
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1),
	target_names=labelNames))
