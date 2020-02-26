from fashion_mnist import fashion_train
import numpy as np
from keras.models import load_model
fashion_model = load_model('fashion_model.h5py')
"""
Plotting accuracy and loss
"""
import matplotlib.pyplot as plt

accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


predicted_labels = fashion_model.predict(test_X)

"""
Since predicted labels are still in matrix representation, we must revert them to their int values
np.argmax does exactly this, returning the index of the highest value in an arrray.
"""
predicted_labels = np.argmax(np.round(predicted_labels), axis=1)
correct = len(np.where(predicted_labels==test_Y)[0])
incorrect = len(np.where(predicted_labels!=test_Y)[0])
print(correct)
print(incorrect)


"""
We need to see which labels the NN is getting incorrect

"""
from sklearn.metrics import classification_report  #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
label_names = ["Label {}".format(i) for i in range(num_classes)]
print(classification_report(test_Y, predicted_labels, target_names=label_names))
"""
             precision    recall  f1-score   support

    Class 0       0.77      0.90      0.83      1000
    Class 1       0.99      0.98      0.99      1000
    Class 2       0.88      0.88      0.88      1000
    Class 3       0.94      0.92      0.93      1000
    Class 4       0.88      0.87      0.88      1000
    Class 5       0.99      0.98      0.98      1000
    Class 6       0.82      0.72      0.77      1000
    Class 7       0.94      0.99      0.97      1000
    Class 8       0.99      0.98      0.99      1000
    Class 9       0.98      0.96      0.97      1000

avg / total       0.92      0.92      0.92     10000
"""
