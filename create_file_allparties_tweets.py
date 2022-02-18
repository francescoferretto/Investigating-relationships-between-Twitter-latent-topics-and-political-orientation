import pandas as pd
import os
from pathlib import Path
import logging
import sys
#
# Configure the path
FILE_PATH = Path(__file__).resolve().parent
DATA_PATH = FILE_PATH / 'data'
OUTPUT_PATH = DATA_PATH / 'tweets_cleaned.csv' 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Create file of tweets of all the parties')

for party in range(1,6):
    logger.info(f'Loading data of party: {party}')
    df = pd.read_csv(DATA_PATH / f'tweets_party{party}_cleaned.csv' )
    df['party'] = party
    
    if party == 1:
        df.to_csv(OUTPUT_PATH)
    else:
        df.to_csv(OUTPUT_PATH, mode='a', header=False)


logger.info(f'Saving the tweets of all the parties in the file: {OUTPUT_PATH}')


