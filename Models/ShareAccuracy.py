from sklearn.metrics import accuracy_score
import numpy as np

def returnAccuracyForNormal(model,Y,X):
    Y_pred = model.predict(X)
    acc=accuracy_score(Y_pred, Y)
    return acc

def accCNN(model,Y,X):
    y_pred = model.predict(X)
    class_labels = np.argmax(y_pred, axis=1)
    Y_test_new = np.argmax(Y, axis=1)
    acc=accuracy_score(class_labels, Y_test_new)
    return acc

