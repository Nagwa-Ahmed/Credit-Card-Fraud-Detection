#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import opendatasets as od

#setting max rows and columns
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

#stop scientific notation
np.set_printoptions(suppress=True)

pd.set_option('display.float_format','{:.20f}'.format)

#download the dataset from kaggle
od.download("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data")
df=pd.read_csv('creditcardfraud/creditcard.csv')

df.head(50)

#descriptive statistics
df.describe()

#check missing values count
df.isna().sum()

"""
Notably, 
* There is no clear missing values. 
* Features have different ranges so they should be scaled
* It's a highly imbalanced problem
* Should detect if there are any outliers
* Will use the whole set of 30 features or only a subset
"""
#Other types of missing data that are not detected in pandas 
#show unique values sorted
for column in df.columns:
    print("Unique values in ",column,"are ",np.sort(df[column].unique()))
    print("------------------------------------------------------------------------------------------------------------------")


#checking the number of duplicates
duplicate_count = df.duplicated().sum()

print(f"Number of duplicate rows: {duplicate_count}")

