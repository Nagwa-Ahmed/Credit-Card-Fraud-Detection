#Imports
import pandas as pd
import opendatasets as od
import matplotlib.pyplot as plt



#download the dataset from kaggle
od.download("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data")
df=pd.read_csv('creditcardfraud/creditcard.csv')

#Percent distribution of fraudulent vs. non-fraudulent transactions
df['Class'].value_counts(normalize=True).plot(kind='barh'
                            ,title='Fraudulent vs. Non-fraudulent Transactions',xlabel='Percentage',
                               ylabel='Class',color=['#a0caec']);
plt.yticks(ticks=[1,0],labels=['Fraudulent','Non-fraudulent'])
plt.text(x=0.36,y=0.5,s='Only 0.172%',fontsize=16,fontstyle='oblique');


#histogram of each variable categorized by class
fraudulent=df[df['Class']==1]#filter the dataset to fraudulent and not
non_fraudulent=df[df['Class']==0]

columns=df.columns
for column in columns:#loop over the columns
    if df.dtypes[column]=='float64': #all columns except for the target which is integer
        
        plt.hist(x=fraudulent[column],label='Fraudulent',color='#f9cd94')#plot a histogram for each feature
        plt.title(f'{column} for fraudulent transactions')
        plt.rcParams['axes.spines.right']=False #set the right and top borders to eliminate clutter
        plt.rcParams['axes.spines.top']=False
        plt.show();
        
        #do the same for non-fraudulent transactions
        plt.hist(x=non_fraudulent[column],label='Non_fraudulent',color='#a0caec')
        plt.title(f'{column} for non-fraudulent transactions')
        plt.rcParams['axes.spines.right']=False
        plt.rcParams['axes.spines.top']=False
        plt.show();


columns=list(columns)
columns.remove('Class')

#Corrletion plots of each pair
#finding all columns combinations
columns_combinations=combinations(columns,2)
#since this iterator can be consumed only once 
# we will convert it to a list

columns_combinations=list(columns_combinations)


#a scatter plot of each combination
for comb in columns_combinations:
    plt.scatter(comb[0],comb[1],data=df,c='#a0caec')
    plt.title(f"{comb[0]} vs. {comb[1]}")
    plt.xlabel(comb[0])
    plt.ylabel(comb[1])
    plt.show();        
    

#It seems that we have few correlations among the features, let's test this correlation

for comb in columns_combinations:
    corr_test=pearsonr(df[comb[0]],df[comb[1]])
    if corr_test[1]<=0.05:
        print(f"{comb[0]} and {comb[1]} are significantly correlated: correlation = {corr_test[0]}")



#Since most of the significant correlations are weak, we will ignore them for now but will keep this in mind





