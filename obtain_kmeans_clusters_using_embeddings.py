import pandas as pd
import os
import logging
from pathlib import Path
import sys
#
# Configure the path
#FILE_PATH = Path(__file__).resolve().parent

FILE_PATH = '/home/emi/unipd/Sartori_CBSD/project/cbsdproject'

DATA_PATH = FILE_PATH + '/data'
DATA_FILE = DATA_PATH + '/tweets_cleaned.csv'
UTILS_PATH = FILE_PATH + '/utils'
MODELS_PATH = FILE_PATH + '/models'

#DATA_PATH = FILE_PATH / 'data'
#DATA_FILE = DATA_PATH / 'tweets_cleaned.csv'
#UTILS_PATH = FILE_PATH / 'utils'
sys.path.append(str(UTILS_PATH))

import clustering_embeddings 


CLASSIFIER_FILE = 'classifier_umlfit_parties_exported.pkl'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Clustering Embeddings')


df = pd.read_csv(DATA_FILE, nrows=300)
party = 1
df_party = df[df['party']==party] 

OUTPUT_EMBEDDINGS_FILE = f'umlfit_embeddings_party{party}.pkl'

clustering_embeddings.get_save_embeddings(CLASSIFIER_FILE, df_party, 'full_text', embeddings_filename=OUTPUT_EMBEDDINGS_FILE)

party=1

OUTPUT_EMBEDDINGS_FILE = f'umlfit_embeddings_party{party}.pkl'

OUTPUT_KMEANS_EMBEDDINGS_FILE = f'labels_kmeanstfidf_party{party}.pkl'

clustering_embeddings.get_clusters(OUTPUT_EMBEDDINGS_FILE, OUTPUT_KMEANS_EMBEDDINGS_FILE)
