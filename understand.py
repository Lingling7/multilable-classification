## -*- Crowdpac_Chenjun Ling -*-

import pandas as pd
import matplotlib.pyplot as plt
import re
import pickle
import numpy

#from ggplot import *
from pandasql import sqldf
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif


######-------1. Understand training data-------######

##---------Instance number
training_set=pd.read_csv('./data/training_set.csv')
        #print(training_set.columns)
print 'training data set instance number: ', len(training_set)

##---------Label number and Visulize the labels
training_set_label= pd.DataFrame(training_set[training_set.columns[1]])
        #print (training_set_label.columns)

# split multiple lables in one row to multiple rows by space
split_labels=pd.concat([pd.Series(row['labels'].split(' '))              
                    for _, row in training_set_label.iterrows()]).reset_index()
split_labels.columns = ['index','labels']

# count labels
pysqldf = lambda q: sqldf(q, globals())
q = """
      select labels, count(labels) as newsnumber from split_labels group by labels order by newsnumber desc;  
    """    
    #Execute SQL command against the pandas frame
lables_count = pysqldf(q)
print lables_count

# visulize label groups
#data=split_labels['labels'].value_counts()
fig, ax = plt.subplots()
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlabel('Labels', fontsize=15)
ax.set_ylabel('Number of news' , fontsize=15)
ax.set_title('Top 5 labels', fontsize=15, fontweight='bold')
lables_count[:5].plot(ax=ax, kind='bar', color='blue')
#plt.title("Distribution of news number under each label")
plt.show()



# count and visulize
#split_labels['Counts'] = split_labels.groupby(['labels']).transform('count')
#print (split_labels)






