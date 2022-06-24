import pickle
import os
from keras.models import model_from_json
import xgboost as xgb

def unPickle(filePath):
    infile = open(filePath, 'rb')
    model = pickle.load(infile)
    return model

def loadModels(datadir,weightFile):
  json_file = open(datadir, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
# load weights into new model
  loaded_model.load_weights(weightFile)
  print("Loaded model from disk")
  return loaded_model
