import numpy as np

def givePredictionNormal(model,X):
    Y_pred=model.predict(X)
    return Y_pred
def giveForCNN(model,X_img):
    Y_pred=model.predict(X_img)
    class_labels = np.argmax(Y_pred, axis=1)
    return class_labels