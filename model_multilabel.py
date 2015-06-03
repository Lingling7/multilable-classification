# -*- Crowdpac_Chenjun Ling -*-

import re
import string
import csv
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import (LinearRegression, RidgeClassifierCV, SGDClassifier,
                                  LogisticRegression)
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_auc_score

training_set=pd.read_csv('./data/training_set.csv')
#training_clean=pd.read_csv('training_clean.csv')
test_set=pd.read_csv('./data/test_set.csv')
#test_clean=pd.read_csv('test_clean.csv')
######-------1. Preprocess-------######

#features_training
    #X-features, text    Y-Labels
X_train = training_set[training_set.columns[0]]
y_train_text=training_set[training_set.columns[1]]

X_test=test_set[test_set.columns[0]].values
y_test_text=test_set[test_set.columns[1]]

    # Binarize the output labels-Y
lb = preprocessing.LabelBinarizer()
Y_train= lb.fit_transform(y_train_text)
Y_test=lb.transform(y_test_text)


######-------3. Set Evaluation-------######


def calculate_result(actual,pred):  
    m_precision = metrics.precision_score(actual,pred)  
    m_recall = metrics.recall_score(actual,pred)
    print 'Hamming_loss:{0:.3f}'.format(hamming_loss(actual, pred, classes=None))
    print 'Precision:{0:.3f}'.format(m_precision)
    print 'Recall:{0:0.3f}'.format(m_recall)  
    print 'F1-score:{0:.3f}'.format(metrics.f1_score(actual,pred,average='micro'))
    
######-------4. Classifier-------######

#4.1 LinearSVC Classifier  
  
print '*************************\n1. LinearSVC \n*************************'  
#LinearSVC uses the One-vs-All (also known as One-vs-Rest) multiclass reduction while SVC uses the One-vs-One multiclass #reduction.
classifier1 = Pipeline([
    ('vectorizer', CountVectorizer()), 
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
fitresult1=classifier1.fit(X_train, Y_train)
predicted1 = classifier1.predict(X_test)
all_labels1 = lb.inverse_transform(predicted1) 
calculate_result(Y_test, predicted1)  
#Compute micro-average ROC curve and ROC area
y_score1 = fitresult1.decision_function(X_test)
average_precision_micro1 = average_precision_score(Y_test, y_score1,average="micro")
print "Average precision:{0:.3f}".format(average_precision_micro1)

#4.2 LinearRegression  
  
print '*************************\n2. LinearRegression\n*************************'  
classifier2 = Pipeline([
    ('vectorizer', CountVectorizer()), 
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearRegression()))])
fitresult2=classifier2.fit(X_train, Y_train)
predicted2 = classifier2.predict(X_test)
all_labels2 = lb.inverse_transform(predicted2) 
calculate_result(Y_test, predicted2) 
#Compute micro-average ROC curve and ROC area
y_score2 = fitresult2.decision_function(X_test)
average_precision_micro2 = average_precision_score(Y_test, y_score2,average="micro")
print "Average precision:{0:.3f}".format(average_precision_micro2)
 
#4.3 LogisticRegression  
  
print '*************************\n3. LogisticRegression\n*************************'  
classifier3 = Pipeline([
    ('vectorizer', CountVectorizer()), 
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LogisticRegression()))])
fitresult3=classifier3.fit(X_train, Y_train)
predicted3 = classifier3.predict(X_test)
all_labels3 = lb.inverse_transform(predicted3) 
calculate_result(Y_test, predicted3) 
#Compute micro-average ROC curve and ROC area
y_score3 = fitresult3.decision_function(X_test)
average_precision_micro3 = average_precision_score(Y_test, y_score3,average="micro")
print "Average precision:{0:.3f}".format(average_precision_micro3)


#4.4 RidgeClassifierCV  
  
print '*************************\n4. RidgeClassifierCV\n*************************'  
classifier4 = Pipeline([
    ('vectorizer', CountVectorizer()), 
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(RidgeClassifierCV()))])
fitresult4=classifier4.fit(X_train, Y_train)
predicted4= classifier4.predict(X_test)
all_labels4 = lb.inverse_transform(predicted4) 
calculate_result(Y_test, predicted4) 
#Compute micro-average ROC curve and ROC area
y_score4 = fitresult4.decision_function(X_test)
average_precision_micro4 = average_precision_score(Y_test, y_score4,average="micro")
print "Average precision:{0:.3f}".format(average_precision_micro4)


#4.5 SGDClassifier  
  
print '*************************\n5. SGDClassifier\n*************************'  
classifier5 = Pipeline([
    ('vectorizer', CountVectorizer()), 
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(SGDClassifier()))])
fitresult5=classifier5.fit(X_train, Y_train)
predicted5= classifier5.predict(X_test)
all_labels5 = lb.inverse_transform(predicted5) 
calculate_result(Y_test, predicted5)
#Compute micro-average ROC curve and ROC area
y_score5 = fitresult5.decision_function(X_test)
average_precision_micro5 = average_precision_score(Y_test, y_score5,average="micro")
print "Average precision:{0:.3f}".format(average_precision_micro5)





######-------4. Print test_set with predict labels-------######
  

#for item, original_label, predit_label in zip(X_test, y_test_text, all_labels):
 #   print '%s, Original: %s, Predict: %s' % (item, original_label, predit_label)

