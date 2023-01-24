# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 11:14:24 2023

@author: mitul
"""
"""
id -             Unique identifier of a customer
gender-          Gender of the customer
area-            Area of the user 
qualification-   Highest Qualification of the customer
income-          Income earned in a year (in rupees)
marital_status-  Marital Status of the customer {0:Single, 1: Married}
Vintage-         No. of years since the first policy date
claim_amount-    Total Amount Claimed by the customer (in rupees)
num_policies-    Total no. of policies issued by the customer
policy-          Active policy of the customer
type_of_policy-  Type of active policy 
cltv-            Customer life time value (Target Variable)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import feature_engine.outliers.winsorizer
from feature_engine.outliers.winsorizer import Winsorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Reading the files
df_train = pd.read_csv("train_BRCpofr.csv")
df_test = pd.read_csv("test_koRSKBP.csv")


#shape of datasets
df_train.shape
df_test.shape

#information of data
df_train.info()
df_test.info()

#checking for null values
df_train.isnull().sum() #no null values
df_test.isnull().sum()  #no null values

#columns datatypes
df_train.dtypes
df_test.dtypes

#describing the data
df_train.describe(include="all")

#dropping
df_train.drop(["id"],axis = 1, inplace = True)

#EDA
#skewness
df_train.skew()
df_train.kurtosis()

#histogram
plt.hist(df_train.claim_amount,color='blue') 
plt.hist(df_train.vintage,color='blue') 
plt.hist(df_train.cltv,color='blue') 

#countplot
sns.countplot(x = "gender", data = df_train)
sns.countplot(x = "area", data = df_train)
sns.countplot(x = "qualification", data = df_train)
sns.countplot(x = "income", data = df_train)
sns.countplot(x = "num_policies", data = df_train)
sns.countplot(x = "policy", data = df_train)
sns.countplot(x = "type_of_policy", data = df_train)

#dropping id column
df_train.drop(["area"],axis = 1, inplace = True)
df_train.drop(["qualification"],axis = 1, inplace = True)
df_train.drop(["marital_status"],axis = 1, inplace = True)
df_train.drop(["gender"],axis = 1, inplace = True)

#there are some features are categorical data so we use label encoding to make categorical data to numerical data
#creating a label encode
labelencode = LabelEncoder()

#performing label encoding on the train dataset
x_df_train = df_train.iloc[:,[0,3,4,5]] #all categorical data
y_df_train = df_train.iloc[:,[1,2,6]] #all numerical data

#making for loop for the label encoding 
for i in x_df_train:
    x_df_train[i] = labelencode.fit_transform(x_df_train[i])
    df1_train = pd.concat([x_df_train, y_df_train],axis=1)


# Checking for outliers:
sns.boxplot(df1_train.income);plt.title('Boxplot');plt.show() #no outliers
sns.boxplot(df1_train.num_policies);plt.title('Boxplot');plt.show() #no outliers
sns.boxplot(df1_train.policy);plt.title('Boxplot');plt.show()  #no outliers
sns.boxplot(df1_train.type_of_policy);plt.title('Boxplot');plt.show() #no outliers
sns.boxplot(df1_train.vintage);plt.title('Boxplot');plt.show()   #no outliers
sns.boxplot(df1_train.claim_amount);plt.title('Boxplot');plt.show() #outliers
sns.boxplot(df1_train.cltv);plt.title('Boxplot');plt.show()  #outliers

#removing the ouliers 
#used the tecniques called the winsorization technique
#treating the outliers in the cltv
# Detection of Outliers
IQR = df1_train['cltv'].quantile(0.75) - df1_train['cltv'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df1_train['cltv'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df1_train['cltv'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

#########################   Winsorization #####################################
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['cltv'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
winsorizer.fit_transform(df1_train[['cltv']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

#treating outliers in the claim_amount
# Detection of Outliers
IQR = df1_train['claim_amount'].quantile(0.75) - df1_train['claim_amount'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = df1_train['claim_amount'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = df1_train['claim_amount'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

######################  Winsorization #####################################
winsorizer = Winsorizer(capping_method ='iqr', # choose skewed for IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5, # 1.5 times of iqr
                          variables=['claim_amount'])
# capping_methods = 'iqr' - 25th quantile & 75th quantile
winsorizer.fit_transform(df1_train[['claim_amount']])

# we can inspect the minimum caps and maximum caps 
winsorizer.left_tail_caps_, winsorizer.right_tail_caps_

#Normalizing the data to make the whole dataset in range of 0 and 1
# or denominator (i.max()-i.min())
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min()) # or denominator (i.max()-i.min())
    return(x)

df1_train_norm = norm_func(df1_train)
df1_train.describe() # min=0, max=1

#correlation using heatmap
sns.heatmap(df1_train_norm.corr())  #no correlation 

#separating the features X(input features) and Y(target feature)
X = df1_train_norm.iloc[:,[0,1,2,3,4,5]] #predictors
Y = df1_train_norm.cltv   #target

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, 
                                                    random_state=0)

# Create an instance of the GradientBoostingRegressor
regressor = GradientBoostingRegressor()

# Train the model on the training set
regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = regressor.predict(X_test)

# Evaluate the performance of the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


#creating the sample submission file
sample_submission = df_test[['id','cltv']]
sample_submission.to_csv("sample_submission.csv",index=False)
