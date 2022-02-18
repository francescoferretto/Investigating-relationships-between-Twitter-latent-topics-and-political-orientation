import csv 
import sys
import json
import pandas as pd
import os
import logging 

# Configure the path 
os.chdir('/home/emi/unipd/Sartori_CBSD/project/cbsdproject')

accounts = pd.read_excel('LabelledAccounts.xlsx', skiprows=1)

import operator
import collections 

def get_party_label(votes, porcentage=0.75):
    porcentage = len(votes)*porcentage
    porcentage = int(porcentage)
    max_party, max_votes = collections.Counter(votes).most_common(1)[0]
    if (max_votes == porcentage):
        return max_party
    else:
        return ''

accounts['party_threshold']=pd.Series(np.zeros())

#Fix a threshold about the political party labelling 
#(we suggest to repeat the analysis with the 100% of agreement of the five judges 
#and with the 75% of agreement of the five judges)
data = pd.read_csv('Tweets/Massimogazza_tweets.csv')
data.head()
