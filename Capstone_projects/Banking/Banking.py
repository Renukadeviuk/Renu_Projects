#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

random_state=42


# Data Exploration

# In[3]:


# Reading data from csv file and examine size of data as well as few rows of data
df = pd.read_csv(r'C:\Users\balaj\Desktop\data1.csv')
print(df.shape)
df.head()


# In[4]:


# Let's get an idea about all the features available in dataset
df.info()


# Here we can see that there are 41 features. Now let's see the statistics of above data:

# In[5]:


# Inspect the mean and standard deviation to see the scale of each features
df.describe()


# In[6]:


# Now check if there is null value in the data!
df.isnull().sum()


# As we can see that, Employment.Type has missing values which we will deal with later.

# Seperating numerical and categorical features

# In[7]:


# List of columns with numerical features
numerical_feature_columns = list(df._get_numeric_data().columns)
numerical_feature_columns


# In[8]:


# List of columns with categorical features
categorical_feature_columns = list(set(df.columns) - set(numerical_feature_columns))
categorical_feature_columns


# Let's plot the histogram of below features to see its distribution

# In[9]:


num_columns = ['disbursed_amount','asset_cost','ltv','PERFORM_CNS.SCORE','PRI.CURRENT.BALANCE','PRI.SANCTIONED.AMOUNT',
            'PRI.DISBURSED.AMOUNT','PRIMARY.INSTAL.AMT','SEC.CURRENT.BALANCE','SEC.SANCTIONED.AMOUNT','SEC.DISBURSED.AMOUNT',
            'SEC.INSTAL.AMT']

for i in range(0, len(num_columns), 2):
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    sns.distplot(df[num_columns[i]], kde=False)
    plt.subplot(122)            
    sns.distplot(df[num_columns[i+1]], kde=False)
    plt.tight_layout()
    plt.show()


# Check categorical data

# In[10]:


df[categorical_feature_columns].head()


# Two features AVERAGE.ACCT.AGE and CREDIT.HISTORY.LENGTH need to convert in terms of years.

# In[11]:


df['AVERAGE.ACCT.AGE'] = df['AVERAGE.ACCT.AGE'].str.replace('yrs ','.',regex=False)
df['AVERAGE.ACCT.AGE'] = df['AVERAGE.ACCT.AGE'].str.replace('mon','',regex=False).astype(float)
df['CREDIT.HISTORY.LENGTH'] = df['CREDIT.HISTORY.LENGTH'].str.replace('yrs ','.',regex=False)
df['CREDIT.HISTORY.LENGTH'] = df['CREDIT.HISTORY.LENGTH'].str.replace('mon','',regex=False).astype(float)
df[categorical_feature_columns].head()


# Now let's examine the Employment.Type feature which has missing values.

# In[12]:


df['Employment.Type'].isnull().sum()


# In[13]:


# Calculate missing percent value from whole dataset
total_null = df.isnull().sum()
percent_null = (total_null/(df.isnull().count())) * 100
missing_data = pd.concat([total_null,percent_null], keys=['Total','Percent'],axis=1)
print(missing_data)


# Missing values contains only 3.285% of the data, hence we can drop it.

# In[14]:


df.dropna(inplace=True)
total_null_1 = df.isnull().sum()
percent_null_1 = (total_null_1/(df.isnull().count())) * 100
missing_data_1 = pd.concat([total_null_1,percent_null_1], keys=['Total','Percent'],axis=1)
print(missing_data_1)


# In[15]:


# Count the each category values from feature
df['Employment.Type'].value_counts()


# In[16]:


# Encode the values in terms of 0 and 1
df['Employment.Type'].replace({'Salaried': 0, 'Self employed': 1}, inplace=True)


# In[17]:


# Dropping unecessary features
df.drop(['Date.of.Birth','DisbursalDate','PERFORM_CNS.SCORE.DESCRIPTION'], axis = 1, inplace=True)


# In[18]:


# Now let's check if null values present in data
df.isnull().sum().sum()


# In[19]:


# Size of the data
df.shape


# In[20]:


# Identify unique values in each features
df.nunique()


# Calculate correlation matrix to inspect correlation among features:

# In[24]:


corr_max = df.corr()  #create correlation matrix
threshold = 0.5
corr_var_list = []
cols = df.columns.tolist()

for i in range(1, len(cols)):
    for j in range(i):
        if((abs(corr_max.iloc[i,j]) > threshold) & (abs(corr_max.iloc[i,j]) < 1)):
            corr_var_list.append([corr_max.iloc[i,j], i, j])

# Sort the list showing higher ones first 
sort_corr_list = sorted(corr_var_list, key=lambda x:abs(x[0]))

#Print correlations and column names
for corr_value, i, j in sort_corr_list:
    print (f"{cols[i]} and {cols[j]} = {round(corr_value, 2)}")


# Plotting distribution of classes of target variable:

# In[29]:


print('Distribution of the loan_default in the dataset')
print(df['loan_default'].value_counts()/len(df))

sns.countplot('loan_default', data=df)
plt.title('Distribution of Classes (Target variable)', fontsize=14)
plt.show()


# In[30]:


# Over sampling to resolve imbalance
df = df.sample(frac=1)
loan_default_1 = df.loc[df['loan_default'] == 1]
loan_default_0 = df.loc[df['loan_default'] == 0]

normal_distributed_df = pd.concat([loan_default_1, loan_default_1, loan_default_1, loan_default_0])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)
new_df.head()


# In[31]:


print('Distribution of the loan_default in the dataset')
print(new_df['loan_default'].value_counts()/len(new_df))

sns.countplot('loan_default', data=new_df)
plt.title('Distribution of Classes (Target variable)', fontsize=14)
plt.show()


# In[32]:


# Size of dataset after over sampling
new_df.shape


# Seperate features and target variable

# In[33]:


X = new_df.drop('loan_default', axis=1)
y = new_df['loan_default'].copy()


# Split train and test data with 70:30 ratio

# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = random_state)


# In[35]:


print("X_train size: ", X_train.shape)
print("X_test size: ", X_test.shape)


# Build and evaluate models
# Define evaluation function which calculates following metrics:
# 
# Confusion matrix
# Accuracy score
# Precision
# Recall
# F1 score
# ROC AUC score.

# In[36]:


def evaluate_model(y_test, y_pred):
    print("Confusion Matrix: \n", metrics.confusion_matrix(y_test, y_pred))
    print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
    print("Precision: ",metrics.precision_score(y_test, y_pred))
    print("Recall: ",metrics.recall_score(y_test, y_pred))
    print("f1 score: ",metrics.f1_score(y_test, y_pred))
    print("roc_auc_score: ",metrics.roc_auc_score(y_test, y_pred))


# Scaling data before model training and testing

# In[37]:


# Scaling training and testing data
scaler = StandardScaler()  
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 1. Logistic Regression¶

# In[40]:


# Find best parameters using grid search
params = {'C':[0.1, 0.5, 1, 5]}

lr = LogisticRegression()
grid = GridSearchCV(estimator=lr, param_grid=params)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
evaluate_model(y_test, y_pred)


# 2. Decision Trees¶

# In[ ]:


params = {'criterion':['gini','entropy'], 'max_depth': [2,3,4,5]}
dt = DecisionTreeClassifier()
dt_clf = GridSearchCV(dt, params)
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
evaluate_model(y_test, y_pred)


# 3. Random Forest

# In[ ]:


rf = RandomForestClassifier(n_estimators=250, random_state=random_state)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
evaluate_model(y_test, y_pred)


# Conclusion
# In this classification problem, it is clear the Random Forest Classifier outperformes Logistic Regression and Decision Trees models.
