#!/usr/bin/env python
# coding: utf-8

# # Loading the required libraries

# In[1]:


# Loading the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE             
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics         #Import scikit-learn metrics module for accuracy calculation
from sklearn.svm import SVC


# # Loading the CSV file into 'input' dataframe

# In[2]:


# The input file is 'COMMA' seperated file and doesn't have header

input = pd.read_csv('C:\\Users\\mahesh\\ML Project\\qsar_oral_toxicity_original.csv',
                    sep=',', header = None)
input.shape


# In[3]:


input.head()


# In[4]:


# Column header is missing in the input file 
# Hence, creating list of dummy column names as c1,c2,...c1023,Label

col = []
for i in range(1024):
    name = 'c'+ str(i)
    col.append(name)
col.append('Label')


# In[5]:


col[1024]


# In[6]:


input.columns = col
input


# In[7]:


input.tail()


# # checking how many unique values are present in each column.

# In[8]:


input.nunique()


#     The above result shows that there are only two values 0,1 in each column

# In[9]:


input.describe()


# In[10]:


# checking whether the dataset is balanced or not
input["Label"].value_counts()


# # Plotting the Label column

# In[11]:


plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
sns.countplot(x='Label',data=input,palette='hls')
plt.savefig('count_plt')
plt.show()


#           Seeing the above result, we can say that the data is not balanced. 
#           Dataset has approx 90% of negative class data,just approx 10% of positive data. 
#           Hence, we have to balance the data before applying the machine learning algorithms.

# # Dropping the "Label" column and storing in the 'y'

# In[12]:


X =  input.drop(['Label'], axis=1)
y = input['Label']
X.shape


# In[13]:


print(y.shape)


# # Counting how many unique values are there in 'Label' column.

# In[14]:


print (input['Label'].value_counts())


# # Counting the occurances of 'Positive' and 'Negative' in the Label column

# In[15]:


def countX(lst, x): 
    count = 0
    for ele in lst: 
        if (ele == x): 
            count = count + 1
    return count 
x = "positive" #0
print('{} has occured {} times'.format(x, countX(y, x)))


# In[16]:


def countX(lst, x): 
    count = 0
    for ele in lst: 
        if (ele == x): 
            count = count + 1
    return count 
x = "negative" #0
print('{} has occured {} times'.format(x, countX(y, x)))


# # Balancing the Imbalanced Dataset

# In[17]:


oversampler = SMOTE()
X_res,y_res = oversampler.fit_resample(X,y)
X_res


# In[18]:


y_res


# # Checking whether oversampled data is balanced or not ?

# In[19]:


x = "positive"
print('{} has occured {} times'.format(x, countX(y_res, x)))
x = "negative"
print('{} has occured {} times'.format(x, countX(y_res, x)))


#          Oversampled data has equal number of positive and negative classes i.e 8251 samples.

# #  Identifying highly correlated columns

# In[20]:


# Create correlation matrix
corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]


# In[21]:


corr_matrix


# In[22]:


upper


# #    Columns that are highly correlated.

# In[23]:


to_drop


# # "X_new" is the new variable storing the data after columns are dropped

# In[24]:


X_new = X_res.drop(['c53','c405','c408','c424','c473','c646','c656',
                    'c657','c683','c685','c758','c759','c819','c871','c960'],axis=1)


# In[25]:


X_new.shape


# # Saving this data for future purpose and re-using. 

# In[26]:


# merging the data and label to single dataframe before saving to csv
df = pd.concat([X_new, y_res], axis=1)


# In[27]:


df


# In[28]:


df.shape


# In[29]:


df.to_csv('C:\\Users\\mahesh\\ML Project\\qsar_oral_toxicity.csv',sep=',')


# # KNN - Implementation

# In[71]:


X_train, X_test, y_train, y_test = train_test_split(X_new, y_res, random_state=1)


# In[72]:


X_train


# In[73]:


print("\nThe shape of the Train data matrix is : ",X_train.shape)


# In[74]:


X_test


# In[75]:


print("\nThe shape of the Test data matrix is :",X_test.shape)


# In[76]:


print("The label train data is:\n",y_train)


# In[77]:


print("The label of test data is:\n",y_test)


# In[34]:


knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)


# In[36]:


y_pred = knn.predict(X_test)


# In[37]:


print("The Confusion matrix is :\n",confusion_matrix(y_test, y_pred))
print("\nThe classification Report is :\n",classification_report(y_test, y_pred))


# In[38]:


#check accuracy of our model on the test data

accuracy = (knn.score(X_test, y_test))*100
print("The Accuracy of our model on the Test data is :",accuracy,"%")


# #  Identifying best k

# In[70]:


import matplotlib.pyplot as plt 

get_ipython().run_line_magic('matplotlib', 'inline')
# choose k between 1 to 31
# k_range = range(1, 31)
k_range = [3,5,7,9,11]
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_new, y_res, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


#     By seeing the above plot for various k values vs accuraries, we can tell that best value of k is 3. As K is increasing, accuracy is decreasing.

#      Now, we are training our Model with the best "K" i.e., 'Three'

# In[39]:


knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)


# In[40]:


#check accuracy of our model on the test data

accuracy_with_best_K = (knn.score(X_test, y_test))*100
print("The Accuracy of our model on the Test data is :",accuracy_with_best_K,"%")


#       Therfore, For QSAR Toxity classification problem, we found that we are getting "85.6% Accuracy" using basic ML model 
#       K-Nearest Neighbourhood.Lets check other ML models

# #    Advantages of K-NN:
#         1. Easy to understand and interpret.
#         2. Easy to implement.
# 
#    

# # Disadvantages of K-NN:
#         1. Model training took almost 3-4 hours for training of such small data for 7 different K values using 5 fold cross validation.
#         2. Training time of 16k records of 1024 dimensions on 16GB Windows took around 3 hours. This concludes that K-NN is not good for high dimensional data.
# 

# # Logistic Regression

# In[66]:


Lg_clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                         class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, 
                         warm_start=False, n_jobs=None, l1_ratio=None)
Lg_clf.fit(X_train, y_train)


# In[42]:


X_train.shape


# In[43]:


X_test.shape


# In[68]:


print("The Confusion matrix is :\n",confusion_matrix(y_test, y_pred))


# In[67]:


print("\nThe classification Report is :\n",classification_report(y_test, y_pred))


# In[45]:


y_pred = Lg_clf.predict(X_test)
Accuracy_with_LogisticRegression = (Lg_clf.score(X_test, y_test))*100
print('Accuracy of logistic regression classifier on test set:',Accuracy_with_LogisticRegression,"%")


# # Support Vector Machine

# In[46]:


svc1 = SVC(kernel='rbf')
svc1.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X_new, y_res, random_state=1)

y_pre = svc1.predict(X_test)


# In[47]:


print("Accuracy:",(metrics.accuracy_score(y_test, y_pre))*100,"%")


#     Observations:
#         1.SVM using RBF Kernel got 96.4% accuracy, which is better than simple K-NN model,
#           slightly complex model Logistic Regression.
#         2.SVM training time is approx 5-7 min, which is more than Logistic Regression.
