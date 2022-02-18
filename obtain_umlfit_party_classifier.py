import pandas as pd
import os
import logging
from pathlib import Path
import sys
#
# Configure the path
FILE_PATH = Path(__file__).resolve().parent
DATA_PATH = FILE_PATH / 'data'
DATA_FILE = DATA_PATH / 'tweets_cleaned.csv'
UTILS_PATH = FILE_PATH / 'utils'
sys.path.append(str(UTILS_PATH))

import umlfit_fastai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Fastai')

logger.info(f'Loading the file of tweets: {DATA_FILE}')

df = pd.read_csv(DATA_PATH / DATA_FILE, nrows=300)

logger.info(df[['full_text', 'party']].head())

logger.info(df['party'].unique())

logger.info(f'Running fastai')

umlfit_fastai.obtain_party_classifier(df, text_column='full_text', label_column='party', substring_filename='umlfit_parties')

