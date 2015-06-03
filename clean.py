#clean text (features) in training and test dataset 
import string
import csv
import pandas as pd
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")
training_set=pd.read_csv('./data/training_set.csv')
test_set=pd.read_csv('./data/test_set.csv')
X_train = training_set[training_set.columns[0]].values
X_test = test_set[test_set.columns[0]].values


# Open File 
resultFile1 = open("./data/training_clean.csv",'wb')
resultFile2 = open("./data/test_clean.csv",'wb')
wr1 = csv.writer(resultFile1, dialect='excel')
wr2 = csv.writer(resultFile2, dialect='excel')
wr1.writerow(["text"])
wr2.writerow(["text"])


# clean text
def clean(s,w):
        data=s
        writer=w
        #remove stop words
        text = ' '.join([word for word in data.split() if word not in stopwords.words("english")])
        #remove digits
        text1 = ''.join([i for i in text if not i.isdigit()])
        
        #remove punctuations
        text2 = text1.translate(string.maketrans("",""), string.punctuation)
        
        #upper to lower
        result=text2.lower()
        
        #write in csv, must use[]
        w.writerow([result])
        return result
        #print 'After: ', result

for i in X_train:
    #print 'Before: ', i    
    clean(i,wr1)        

for j in X_test:
    clean(j,wr2)

