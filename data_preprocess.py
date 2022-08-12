#pip install scikit-learn
from sklearn.model_selection import train_test_split
from feature_engineering import feature_engineering
import numpy as np
import pandas as pd
def data_preprocess():
    data = pd.read_csv('bank.csv')
    balanced_data = feature_engineering()
    X = balanced_data.drop(['y_new'],axis=1)
    y = balanced_data['y_new']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)
    print(len(X_train),len(X_test))

data_preprocess()    
