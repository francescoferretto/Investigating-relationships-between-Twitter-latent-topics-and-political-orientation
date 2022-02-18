import csv 
import sys
import json
import pandas as pd
import os
import logging 
import collections
import re
import numpy as np
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Cleaning users accounts collected from Twitter')

# Configure the path 
os.chdir('/home/emi/unipd/Sartori_CBSD/project/cbsdproject')

accounts = pd.read_excel('data/LabelledAccounts.xlsx', skiprows=1)

# Unify all the unknow gender with the letter U
accounts['Sex (M/F)'].replace(['-', '?', 'D', '\n', 'U\n', 'C', 'P', 'U ', '.', 'F?'], 'U', inplace=True)
# Unify gender Female
accounts['Sex (M/F)'].replace(['M ', 'M\n', '\nM'], 'M', inplace=True)
# Unify gender Male
accounts['Sex (M/F)'].replace(['F ', 'F\n', '\nF'], 'F', inplace=True)
# Fill nan values with U
accounts['Sex (M/F)'].fillna('U', inplace=True)

# there are a lot of null in SID 5 because the are groups with only 4 students.

# then in gender i have the values F, M and 0 because i dont know


accounts['SID 2'].replace(['?', '0'],0, inplace=True)
accounts['SID 4'].replace(['?'],0, inplace=True)

logger.info('Obtaining the party with max votes per user')

accounts.dropna(how='all', inplace=True, subset=['SID 1', 'SID 2', 'SID 3', 'SID 4', 'SID 5'])
arr = accounts[['SID 1', 'SID 2', 'SID 3', 'SID 4', 'SID 5']]
max_val = arr.mode(axis=1).iloc[:, 0].astype(int)
count_max = (arr.apply(lambda x: x.value_counts(normalize=True), axis=1)
             .fillna(0)
             .max(axis=1))
accounts1 = accounts.assign(Max_Value = max_val, Threshold = count_max)
accounts1.dropna(subset=['Max_Value'], inplace=True)


# Remove @ symbol at the beginning of some values in the column Twitter ID
accounts1['Twitter ID'] = accounts['Twitter ID'].str.replace('@','')
# No repeated users
accounts1.drop_duplicates(subset=['Twitter ID'], inplace=True)

# Count how many users have tweets
usernames = accounts1['Twitter ID']
usernames = usernames.values

def number_of_lines(file):
    with open(file) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

logger.info('Checking accounts who have the tweets files')

def get_validate_file(username):
    if os.path.exists('Tweets/%s_tweets.csv' % (username)):
        return 'Tweets/%s_tweets.csv' % (username)
    elif os.path.exists('Tweets/@%s_tweets.csv' % (username)):
        return 'Tweets/@%s_tweets.csv' % (username)
    else:
        return None

valid_usernames = dict()
for username in usernames:
    file = get_validate_file(username)
    if file is not None:
        n_tweets = number_of_lines(file)
        valid_usernames[username]=n_tweets
          

logger.info('Saving the validated accounts in a csv file')

valid_usernames_df=pd.DataFrame(columns=['Twitter ID', 'Tweets'])
valid_usernames_df['Tweets'] = valid_usernames.values()  
valid_usernames_df['Twitter ID'] = valid_usernames.keys()
valid_accounts = pd.merge(valid_usernames_df, accounts1, on='Twitter ID', how='inner') 
valid_accounts.to_csv('data/valid_accounts_party_labeled.csv')