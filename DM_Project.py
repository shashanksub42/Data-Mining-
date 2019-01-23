
# coding: utf-8

# In[49]:

import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, RFE
import scipy.stats as stats
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[82]:

#Load data 
inp_data = pd.read_csv("C:/Users/shash/Desktop/UCF/Fall 2018/Data Mining 1/Project/DataSet/UCF Dataset 2018 - Training set.csv")


# In[83]:

inp_data.head()


# In[52]:

obj_df = inp_data.select_dtypes(include=['object']).copy()
obj_df[obj_df.isnull().any(axis=1)]
obj_df = obj_df.fillna({'OccupancyStatus' : 'RENT'})
obj_df = obj_df.fillna({'RequestType' : 'INDIRECT'})


# In[53]:

#Encode categorical variables
lb_make = LabelEncoder()
cols = obj_df.columns
orig_cols = inp_data.columns
for cols in cols:
    obj_df[cols] = lb_make.fit_transform(obj_df[cols])
    inp_data[cols] = obj_df[cols]


# In[54]:

inp_data.isnull().sum()
inp_data = inp_data.fillna(inp_data['CoMonthlyLiability'].value_counts().index[0])
inp_data = inp_data.fillna(inp_data['CoMonthlyRent'].value_counts().index[0])
inp_data = inp_data.fillna(inp_data['Loanterm'].value_counts().index[0])
inp_data = inp_data.fillna(inp_data['EstimatedMonthlyPayment'].value_counts().index[0])
inp_data = inp_data.fillna(inp_data['NumberOfOpenRevolvingAccounts'].value_counts().index[0])
inp_data = inp_data.fillna(inp_data['LTV'].value_counts().index[0])
inp_data = inp_data.fillna(inp_data['DTI'].value_counts().index[0])


# In[57]:

temp_df = inp_data
temp_df.drop(['LoanStatus'], axis = 1)


# In[59]:

# Feature Extraction with RFE
# load data
X = temp_df.drop(['LoanStatus'], axis = 1)
Y = inp_data['LoanStatus']
# feature extraction
model = LogisticRegression()
rfe = RFE(model, 25)
fit = rfe.fit(X, Y)
print("Num Features: ", fit.n_features_)
print("Selected Features: ", fit.support_)
print("Feature Ranking: ", fit.ranking_)


# In[60]:

#Drop LoanStatus from list of columns
temp_arr = np.array(temp_df.drop(['LoanStatus'], axis = 1).columns)

#Append highest ranking features into 'arr'
arr = list()
for var in range(0, len(fit.ranking_)):
    if(fit.ranking_[var] == 1):
        arr.append(temp_arr[var])


# In[61]:

drop_list = list(set(temp_df.drop(['LoanStatus'], axis = 1).columns) - set(arr))
cols = [col for col in temp_df.columns if col not in drop_list]
X_df = temp_df[cols]
X_df.drop(['LoanStatus'], axis = 1, inplace = True)
X_df.shape


# In[71]:

X_df.head()


# In[62]:

from sklearn.model_selection import  GridSearchCV, train_test_split


# In[63]:

X_train, X_test, y_train, y_test = train_test_split(X_df, Y, test_size = 0.3)


# In[64]:

from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)


# In[65]:

from sklearn.metrics import confusion_matrix, f1_score


# In[77]:

from sklearn.ensemble import RandomForestClassifier
clf_rand = RandomForestClassifier(n_estimators=1000,
                             random_state=0)
clf_rand = clf_rand.fit(X_train, y_train)


# In[78]:

estimator_limited = clf_rand.estimators_[5]


# In[67]:

pred_rand = clf_rand.predict(X_test)


# In[68]:

from sklearn.metrics import confusion_matrix, f1_score


# In[69]:

confusion_matrix(y_true=y_test, y_pred=pred_rand)


# In[70]:

f1_score(y_test, pred_rand)


# In[74]:







