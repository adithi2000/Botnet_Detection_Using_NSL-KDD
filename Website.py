import streamlit
import streamlit as st
from TestEssentials import TestingFile
import pandas as pd

def processWeb():
    siteHeader = st.container()
    dataExploration = st.container()
    Features = st.container()
    modelDescription = st.container()
    modelTesting = st.container()
    with siteHeader:
        st.title('Welcome to multilayer framework for Botnet Detection')
        st.text('This Project is used to find Botnet attack via Clustering and Classification of data')
    with dataExploration:
        st.header('Dataset: NSL KDD Dataset')
        st.text('The Dataset is as follows:')
        data = pd.read_csv('TestData/KDD Train+ Unwanted removed.txt')
        columns = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
        'root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
        'count','srv_count', 'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate', 'srv_diff_host_rate','dst_host_count',
        'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'class'
    ]
        data.columns = columns
        st.write(data.head(10))
        st.write('The NSL KDD Dataset NSL-KDD is a new version data set of the KDD\'99 data set.'
                 ' This is an effective benchmark data set to help researchers compare different intrusion detection methods.')
        st.write("Furthermore, the number of records in the NSL-KDD train and test sets are reasonable. This advantage makes it affordable to run the experiments on the complete set without the need to randomly select a small portion. "
                 "Consequently, evaluation results of different research work will be consistent and comparable.")
        st.write("The dataset can be collected from here https://www.unb.ca/cic/datasets/nsl.html")

        st.subheader("The Data looks this way when given to CNN model")
        st.image('TestData/img.png')

    with Features:
        st.header("Features of the Dataset:")
        st.write("The Following Features were Selected Among the Given Features of the Dataset by the Method of Using Feature "
                 "Importances From the XGBoost,Logisitic Regression and Support Vector Machines Models")
        st.write("The Details of the features selected is as follows: ")
        st.write("Important Descritpion of the Features can be viewed at \n"
                 "https://e-tarjome.com/storage/btn_uploaded/2019-07-13/1563006133_9702-etarjome-English.pdf")
        st.write("* **The features dropped are:** ")
        st.write(['land', 'urgent', 'num_failed_logins', 'root_shell', 'su_attempted', 'num_root', 'num_shells',
                   'num_access_files', 'num_outbound_cmds', 'is_host_login', 'serror_rate', 'srv_rerror_rate'])

    with modelDescription:
        st.header("Model Description:")
        st.write("The Description of Manual Analysis of Cluster can be found at the following link below:")
        st.write("https://docs.google.com/spreadsheets/d/1S1GZrGUzhrOXmDEcmNsASqw20eGKYjJo/edit?usp=sharing&ouid=111324702527640055586&rtpof=true&sd=true")
        st.subheader("Silhouette Analysis of Clusters")
        st.write("There are 4 basic clusters formed. The Kmeans Cluster were choosen based on Elbow method on the training data")
        st.write("More information on Silhoutte can be viewed at https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html")
        st.image('TestData/img_1.png')
        st.write("Silhouette Score for 4 clusters 0.4201")

        st.subheader("Types of Models for Classification")
        st.write("After clustering the cluster output is used for classification")
        st.write("The Algorithms used for Classification are")
        st.markdown("* **XGBoost**")
        st.markdown("* **Logistic Regression**")
        st.markdown("* **Support Vector Machine**")
        st.markdown("* **Convolutional Neural Network**")

        st.subheader("Information on cluster(Manually Analysed)")
        st.write("The follwing type of attacks were analysed based on the attack description and few necessary features available from various sources:")
        st.markdown("* **Cluster 0 :** Probe/DoS")
        st.markdown("* **Cluster 1 :**  Normal/DoS")
        st.markdown("* **Cluster 2 :** DoS/R2L/U2R")
        st.markdown("* **Cluster 3 :** U2R/R2L")

    with modelTesting:
        st.header("Upload you dataset to check the prediction")
        data_file = st.file_uploader("Upload CSV", type=['csv','txt'])
        if data_file is not None:
            file_details = {"Filename": data_file.name, "FileType": data_file.type, "FileSize": data_file.size}
            st.write(file_details)
            prediction=TestingFile.testing(data_file)
            st.write("Prediction from XGBoost is: ",prediction['XGB_Prediction'])
            st.write("* **Accuracy :**",prediction['XGBACC'])
            st.write("Prediction from Logisitic Regression is: ", prediction['LR_Prediction'])
            st.write("* **Accuracy :**", prediction['LRACC'])
            st.write("Prediction from SVM is: ", prediction['SVM_Prediction'])
            st.write("* **Accuracy :**", prediction['SVMACC'])
            st.write("Prediction from CNN is: ", prediction['CNN_Prediction'])
            st.write("* **Accuracy :**", prediction['CNNACC'])

if __name__=='__main__':
    processWeb()

