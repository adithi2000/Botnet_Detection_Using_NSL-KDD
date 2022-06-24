from Preprocessing import DataPreProcessing as dp
from xgboost import XGBClassifier as xgb
import pickle
def training(filename):
    #"TestData\\KDDTest+ Unwanted Data Removed.txt"
    X_train=dp.DataPreProcess(filename)
    Y_train=dp.get_Y("../PickleFiles/kmeans_cluster.pkl", X_train)
    xgb_multi = xgb(n_estimators=200, max_depth=5)
    xgb_multi.fit(X_train, Y_train)
    pickle.dump(xgb_multi, open("../PickleFiles/xgboost_kmeans.pkl", "wb"))
if __name__=="__main__":
    pred=training("../TestData/KDD Train+ Unwanted removed.txt")
