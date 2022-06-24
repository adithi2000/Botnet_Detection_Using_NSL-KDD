import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle

def Standardization(data):
  from sklearn.preprocessing import StandardScaler
  scaler1 = StandardScaler().fit(data)
  X_original=scaler1.transform(data)
  return X_original

def DataPreProcess(filename):
    columns = [
        'duration',
        'protocol_type',
        'service',
        'flag',
        'src_bytes',
        'dst_bytes',
        'land',
        'wrong_fragment',
        'urgent',
        'hot',
        'num_failed_logins',
        'logged_in',
        'num_compromised',
        'root_shell',
        'su_attempted',
        'num_root',
        'num_file_creations',
        'num_shells',
        'num_access_files',
        'num_outbound_cmds',
        'is_host_login',
        'is_guest_login',
        'count',
        'srv_count',
        'serror_rate',
        'srv_serror_rate',
        'rerror_rate',
        'srv_rerror_rate',
        'same_srv_rate',
        'diff_srv_rate',
        'srv_diff_host_rate',
        'dst_host_count',
        'dst_host_srv_count',
        'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate',
        'dst_host_srv_serror_rate',
        'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
        'class'
    ]
    #data = pd.read_csv("KDD Train+ Unwanted removed.txt")
    data = pd.read_csv(filename)
    data.columns = columns
    cat_col = ['protocol_type', 'service', 'flag', 'class']
    new_categorical_columns = data[cat_col]
    new_categorical_columns.head()
    new_cat_encoded = new_categorical_columns.apply(LabelEncoder().fit_transform)
    data = data.drop(['flag', 'protocol_type', 'service'], axis=1)
    data = data.drop('class', axis=1)
    data = data.join(new_cat_encoded)
    data = data.reindex(columns, axis=1)
    X = data.drop('class', 1)
    Y = data['class']
    X = data.drop(['land', 'urgent', 'num_failed_logins', 'root_shell', 'su_attempted', 'num_root', 'num_shells',
                   'num_access_files', 'num_outbound_cmds', 'is_host_login', 'serror_rate', 'srv_rerror_rate'], 1)
    X_original = Standardization(X)
    return X_original

def get_Y(filePath,data):
    infile = open(filePath, 'rb')
    kmeans = pickle.load(infile)
    Y_train_new = kmeans.predict(data)
    return Y_train_new