#!/usr/bin/env python
# coding: utf-8

# _**Factors Respnsible for Heart ❤ Attack Using Machine Learning- Predicting by 
# Logistic Regression & 
# Random Forest Classifier**_

# # Data Wrangling
# 
# 

# In[ ]:


#Necessary Imports

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[1]:


#Read the file to using pandas(pd)

data =pd.read_excel('/content/data.xlsx')

data.head(5)


# In[ ]:


# Read the columns, Rows and get the shape of the two-dimensional dataframe

print("(Rows, columns): " + str(data.shape))
data.columns


# In[ ]:


data.nunique(axis=0)# returns the number of unique values for each variable.


# In[ ]:


#summarizes the count, mean, standard deviation, min, and max for numeric variables.
data.describe()


# In[ ]:


# Display the Missing Values

print(data.isna().sum())


# _No null values identified, pretty clean data!_

# Lets see if theirs a good proportion between our positive & negative binary predictor.

# In[ ]:


data['target'].value_counts()


# **Correlation Matrix**- let’s you see correlations between all variables.
# 
# Within seconds, you can see whether something is positively or negatively correlated with our predictor (target).

# In[ ]:


# calculate correlation matrix

corr = data.corr()
plt.subplots(figsize=(15,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
sns.heatmap(corr, xticklabels=corr.columns,
            yticklabels=corr.columns, 
            annot=True,
            cmap=sns.diverging_palette(220, 20, as_cmap=True))


# *We can see there is a positive correlation between chest pain (cp) & target (our predictor). This makes sense since, the greater amount of chest pain results in a greater chance of having heart disease. Cp (chest pain), is a ordinal feature with 4 values: Value 1: typical angina ,Value 2: atypical angina, Value 3: non-anginal pain , Value 4: asymptomatic.*
# 
# *In addition, we see a negative correlation between exercise induced angina (exang) & our predictor. This makes sense because when you excercise, your heart requires more blood, but narrowed arteries slow down blood flow.*

# In[ ]:


# Using pairplots to see the continuous columns variable correlation

subData = data[['age','trestbps','chol','thalach','oldpeak']]
sns.pairplot(subData)


# Making a smaller pairplot with only the continuous variables, to dive deeper into the relationships. Also a great way to see if theirs a positive or negative correlation!

# In[ ]:


sns.catplot(x="target", y="oldpeak", hue="slope", kind="bar", data=data);

plt.title('ST depression (induced by exercise relative to rest) vs. Heart Disease',size=25)
plt.xlabel('Heart Disease',size=20)
plt.ylabel('ST depression',size=20)


# *ST segment depression occurs because when the ventricle is at rest and therefore repolarized. If the trace in the ST segment is abnormally low below the baseline, this can lead to this Heart Disease. This supports the plot above because low ST Depression yields people at* *greater risk for heart disease*. While a high ST depression is considered normal & healthy. The “slope” hue, refers to the peak exercise ST segment, with values: 0: upsloping , 1: flat , 2: downsloping). Both positive & negative heart disease patients exhibit equal distributions of the 3 slope categories.*

# **Outlier detection Using box plot and violin plot**

# In[ ]:


# Violin plot


plt.figure(figsize=(12,8))
sns.violinplot(x= 'target', y= 'oldpeak',hue="sex", inner='quartile',data= data )
plt.title("Thalach Level vs. Heart Disease",fontsize=20)
plt.xlabel("Heart Disease Target", fontsize=16)
plt.ylabel("Thalach Level", fontsize=16)


# *We can see that the overall shape & distribution for negative & positive patients differ vastly. Positive patients exhibit a lower median for ST depression level & thus a great distribution of their data is between 0 & 2, while negative patients are between 1 & 3. In addition, we don’t see many differences between male & female target outcomes.*

# In[ ]:


# Box Plot

plt.figure(figsize=(12,8))
sns.boxplot(x= 'target', y= 'thalach',hue="sex", data=data )
plt.title("ST depression Level vs. Heart Disease", fontsize=20)
plt.xlabel("Heart Disease Target",fontsize=16)
plt.ylabel("ST depression induced by exercise relative to rest", fontsize=16)


# *Positive patients exhibit a heightened median for ST depression level, while negative patients have lower levels. In addition, we don’t see many differences between male & female target outcomes, expect for the fact that males have slightly larger ranges of ST Depression.*

# **Filtering data by positive & negative Heart Disease patient**

# In[ ]:


# Filtering data by POSITIVE Heart Disease patient
pos_data = data[data['target']==1]
pos_data.describe()


# In[ ]:


# Filtering data by NEGATIVE Heart Disease patient
pos_data = data[data['target']==0]
pos_data.describe()


# In[ ]:


# Filtering data by NEGATIVE Heart Disease patient
neg_data = data[data['target']==0]
neg_data.describe()


# In[ ]:


print("(Positive Patients ST depression): " + str(pos_data['oldpeak'].mean()))
print("(Negative Patients ST depression): " + str(neg_data['oldpeak'].mean()))


# In[ ]:


print("(Positive Patients thalach): " + str(pos_data['thalach'].mean()))
print("(Negative Patients thalach): " + str(neg_data['thalach'].mean()))


# *From comparing positive and negative patients we can see there are vast differences in means for many of our 13 Features. From examining the details, we can observe that positive patients experience heightened maximum heart rate achieved (thalach) average. In addition, positive patients exhibit about 1/3rd the amount of ST depression induced by exercise relative to rest (oldpeak).*

# # **Machine Learning + Predictive Analytics**

# Prepare Data for Modeling
# 
# To prepare data for modeling, just remember ASN (Assign,Split, Normalize).
# 
# Assign the 13 features to X, & the last column to our classification predictor, y

# In[ ]:


X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# In[ ]:


# Split: the data set into the Training set and Test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)


# In[ ]:


#Normalize: Standardizing the data will transform the data so that its distribution will have a 
#mean of 0 and a standard deviation of 1

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[ ]:


# Prediction using the Classification model Logistic Regression

from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(random_state=1) # get instance of model
model1.fit(x_train, y_train) # Train/Fit model 

y_pred1 = model1.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred1)) # output accuracy


# *Achieved 74% accuracy*

# In[ ]:


# Prediction using Random Forest Classifier

from sklearn.metrics import classification_report 
from sklearn.ensemble import RandomForestClassifier

model6 = RandomForestClassifier(random_state=1)# get instance of model
model6.fit(x_train, y_train) # Train/Fit model 

y_pred6 = model6.predict(x_test) # get y predictions
print(classification_report(y_test, y_pred6)) # output accuracy


# *Achieved 80% accuracy*

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred6)
print(cm)
accuracy_score(y_test, y_pred6)


# *Thus, from confusion matrix we conclude a good outcome as 80% is the ideal accuracy!*

# In[ ]:




