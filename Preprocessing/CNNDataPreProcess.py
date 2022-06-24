import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
def convertToImage(X_original):
  x=list()
  count=0
  for i in X_original:
    array = np.reshape(i, (5,6))
    print(count)
    count+=1
    img=np.reshape(array,(array.shape[0],array.shape[1],1))
    x.append(img)
  X=np.array(x)
  return X
def get_Y_categorical(data_Y):
  Y_train_new = to_categorical(data_Y)
  return Y_train_new


