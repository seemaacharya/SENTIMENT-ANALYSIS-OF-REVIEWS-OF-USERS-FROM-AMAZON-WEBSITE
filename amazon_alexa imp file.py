# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 21:19:28 2021

@author: DELL
"""
#Project-Alexa reviews
#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset
alexa=pd.read_csv("amazon_alexa dataset.csv")
alexa.head()

#EDA
alexa.describe()
alexa.groupby("rating").count()
#Here we can see that rating with 2 star is having the least reviews i.e 96 reviews whereas
#rating with 5 stars are the highest number of reviews i.e 2286
alexa.groupby("feedback").count()
#The feedback with 0 is having only 257 reviews whereas feedback with 1 have 2893 reviews.
#That means the dataset contains 10 times positive reviews compared to the negative reviews.

alexa.query('feedback=="0" & rating>2')
#The feedback with 0 are less than 2 star rating as the feedback with 0 greater than rating 2 is not showing any entries

alexa[alexa['feedback']==0]['verified_reviews'].iloc[2]


#To check the length of the reviews
alexa["length"]=alexa["verified_reviews"].apply(len)
alexa["length"].head()

alexa.length.max()
#2851-this is the longest review with 2851 characters
alexa_max=alexa.query('length=="2851"')

alexa.length.min()
alexa.query('length=="1"')
alexa.query('length=="2"')
alexa.query('length=="3"')


#Visualization
%matplotlib inline
alexa['length'].plot(bins=50, kind='hist')
#Majority of the reviews are the reviews ranging b/w 0 and 200 characters

#Cleaning and text data preparation
#cleaning
import re
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


corpus = []

for i in range(0, 3150):
    review = re.sub('[^a-zA-Z]', ' ', alexa['verified_reviews'][i])
    review = review.lower()
    review = review.split()
    #ps = PorterStemmer()
    lm=WordNetLemmatizer()
    #review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = [lm.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
   
#Bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec=TfidfVectorizer() 
X = tfidf_vec.fit_transform(corpus).toarray()
y= alexa.iloc[:,4].values

print(X.shape)
#(3150, 3479)
print(y.shape)
#(3150,)
 
#word cloud
import os
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
#create and generate wordcloud image
wordcloud=WordCloud().generate(str(corpus))
#display the generated image
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
#we see alexa, love, great,use are all positive words and these words are mostly used  by the users.


#train_test split
from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


#Normalize the data
from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
x_train=mm.fit_transform(x_train)
x_test=mm.fit_transform(x_test)



#Model building
#building model with Naive Bayes 
from sklearn.naive_bayes import MultinomialNB
model_mnb=MultinomialNB()
model_mnb.fit(x_train,y_train)
model_mnb.score(x_train,y_train)
#train accuracy=96.37
pred_mnb=model_mnb.predict(x_test)

#Accuracy using confusion matrix for MultinomialNB
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred_mnb))
#92.169%
pd.crosstab(y_test,pred_mnb)
#we see that majority class has been correctly classified as 
#847/870=97%
#minority class is 24/75
#24+51=32%
#From this we can figure out that as our dataset is imbalanced so accuracy is biased towards majority class.

#Hence we will use SMOTE technique to balance the dataset.
#BALANCING THE DATASET
pip install imblearn
from imblearn.over_sampling import SMOTE
smote = SMOTE()
x_train_smote, y_train_smote = smote.fit_resample(x_train.astype('float'),y_train)

from collections import Counter
print("Before SMOTE :" , Counter(y_train))
#Before SMOTE : Counter({1: 2023, 0: 182})
print("After SMOTE :" , Counter(y_train_smote))
#After SMOTE : Counter({1: 2023, 0: 2023})

#Again model building and checking the accuracy after balancing the dataset
model_mnb.fit(x_train_smote,y_train_smote)
pred_mnb=model_mnb.predict(x_test)
print(accuracy_score(y_test,pred_mnb))
#88.57
pd.crosstab(y_test,pred_mnb)
#here majority class
#786+84
#786/870=90%
#minority class
#51+24
#51/75=68%
#Accuracy has now improved a lot after applying SMOTE

#COMPARISON
#In case of imbalanced dataset= Before applying SMOTE accuracy for majority class(for group 1 in y) was 97%,
#minority class(for group 0 in y)was only 32% 
#In case of balanced dataset=majority class accu=90%, minority accu=68%





#We wil try building another models as well.

#RANDOM FOREST
#Building RandomForest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix
model2=RandomForestClassifier()
model2.fit(x_train,y_train)
y_pred=model2.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
#94.07%
pd.crosstab(y_test,pred_mnb)
#majority=786+84=870
#786/870=90.34%
#minority=51+24=75
#51/75=68%


#Applying SMOTE(same process as before)
smote = SMOTE()
x_train_smote, y_train_smote = smote.fit_resample(x_train.astype('float'),y_train)

from collections import Counter
print("Before SMOTE :" , Counter(y_train))
#Before SMOTE : Counter({1: 2023, 0: 182})
print("After SMOTE :" , Counter(y_train_smote))
#After SMOTE : Counter({1: 2023, 0: 2023})

model2.fit(x_train_smote,y_train_smote)
y_pred=model2.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
#91%
pd.crosstab(y_test,pred_mnb)
786/870
#majority=786+84=870
#786/870=90.34%(same as before)
#minority=51+24=75
#51/75=68%
#Now our dataset is balanced

#model building on balanced dataset
from sklearn.ensemble import RandomForestClassifier
model_rf=RandomForestClassifier()
from sklearn.model_selection import KFold, cross_val_score
num_trees=100
kfold = KFold(n_splits=8, random_state=None)
model_rf = RandomForestClassifier(n_estimators=num_trees)
model_rf.fit(x_train_smote,y_train_smote)
pred_new=model_rf.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,pred_new))
#91%
pd.crosstab(y_test,pred_new)
#majority=820/870=94%(improved)
#minority=41/75=54.66%

