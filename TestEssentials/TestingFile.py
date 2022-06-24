from Models import ImportModels as Im
from Models import ShareAccuracy as SA
from Models import ShowPrediction as ShP

from Preprocessing import CNNDataPreProcess as cnnDp
from Preprocessing import DataPreProcessing as dp

'''def GetAccuracyForTestData(model,data,Y):
    accuracy=list()
    accuracy.append(SA.returnAccuracyForNormal(model,Y,data),SA.accCNN(model,Y,data))
    return accuracy'''

def testing(filename):
    #"TestData\\KDDTest+ Unwanted Data Removed.txt"
    X_test=dp.DataPreProcess(filename)
    Y_test=dp.get_Y("PickleFiles/kmeans_cluster.pkl", X_test)
    X_test_img=cnnDp.convertToImage(X_test)
    Y_test_cat=cnnDp.get_Y_categorical(Y_test)

    #testing with Normal Models
    #xgboost,lr and svm
    XGB=Im.unPickle("PickleFiles/xgboost_kmeans.pkl")
    LR=Im.unPickle("PickleFiles/LR_kmeans.pkl")
    SVM=Im.unPickle("PickleFiles/svm_kmeans.pkl")

    #Import the CNN Model:
    CNN=Im.loadModels("PickleFiles/CNN_kmeans.json","PickleFiles/CNN_kmeans.h5")

    prediction_dict={}
    prediction_dict["XGB_Prediction"]=ShP.givePredictionNormal(XGB,X_test)
    prediction_dict["LR_Prediction"] = ShP.givePredictionNormal(LR,X_test)
    prediction_dict["SVM_Prediction"] = ShP.givePredictionNormal(SVM,X_test)

    prediction_dict["CNN_Prediction"]=ShP.giveForCNN(CNN,X_test_img)

    prediction_dict["XGBACC"]=SA.returnAccuracyForNormal(XGB,Y_test,X_test)
    prediction_dict["LRACC"] = SA.returnAccuracyForNormal(LR,Y_test,X_test)
    prediction_dict["SVMACC"] = SA.returnAccuracyForNormal(SVM,Y_test,X_test)
    prediction_dict["CNNACC"] = SA.accCNN(CNN,Y_test_cat,X_test_img)
    return prediction_dict
'''if __name__=="__main__":
    pred=testing("/TestData/KDDTest+ Unwanted Data Removed.txt")
    print(pred)'''