# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 10:51:52 2019

@author: Nat, Emi
"""
import pandas as pd
import os
from pathlib import Path
import sys
#
# Configure the path
FILE_PATH = Path(__file__).resolve().parent
DATA_PATH = FILE_PATH / 'data'
TWEETS_PATH = FILE_PATH / 'Tweets'
UTILS_PATH = FILE_PATH / 'utils'
sys.path.append(str(UTILS_PATH))
import text_processing as tp

accounts = pd.read_csv(DATA_PATH / 'valid_accounts_party_labeled.csv')

accounts = accounts[accounts['Max_Value'] != 0]
accounts["Max_Value"] = accounts["Max_Value"].astype(int)
parties = list(accounts['Max_Value'].unique())
stopwords=tp.get_stopwords('it')

#Filter the accounts with agreement 100%
accounts = accounts[accounts['Threshold']==1.00]

for party in parties:
    print(f'party:{party}')
    OUTPUT_PATH = DATA_PATH / f'tweets_party{party}_cleaned.csv'
    accounts_names = accounts[accounts['Max_Value'] ==
                              party]['Twitter ID'].values
    print('cleaning all the tweets using text mining')
    for i, account in enumerate(accounts_names):
        file = TWEETS_PATH / f'{account}_tweets.csv'
        if os.path.exists(file):
            df1 = pd.read_csv(file)
            df1['full_text'] = df1['full_text'].str.replace('\r', '')

            df1['full_text_cleaned'] = df1['full_text'].apply(tp.clean_sentence, join=True, stop_words=stopwords)
            df1.dropna(inplace=True)
            df1 = df1[df1["full_text_cleaned"] != ""]

            if i == 0:
                df1.to_csv(OUTPUT_PATH)
            else:
                df1.to_csv(OUTPUT_PATH, mode='a', header=False)