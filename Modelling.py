#Imports
import pandas as pd
#import opendatasets as od
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump,load

#download the dataset from kaggle
#od.download("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data")
df=pd.read_csv('creditcardfraud/creditcard.csv')



columns=df.columns
columns=list(columns)
columns.remove('Class')

X=df[columns]
y=df['Class']


#function to split data to train,validation and test
def train_valid_test(X,y,valid,test):
    if not 0<valid<1 or not 0<test<1:#valid and test are the percentages of validation and test 
        raise ValueError('valid and test should be between 0 and 1')
    #first hold out the test set    
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test,random_state=42, stratify=y)
    
    #then split to train and validation
    X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=(valid/(1-test)),random_state=42,stratify=y_train)
    
    return X_train,X_val,X_test,y_train,y_val,y_test



X_train,X_val,X_test,y_train,y_val,y_test=train_valid_test(X,y,0.2,0.2)

rf=RandomForestClassifier(random_state=42,class_weight={1:0.5,0:0.5})

rf.fit(X_train,y_train)

y_train_pred = rf.predict(X_train)
y_val_pred = rf.predict(X_val)

#saving the model

dump(rf,'rf.joblib')