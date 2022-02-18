import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
#
# Configure the path
FILE_PATH = Path(__file__).resolve().parent
DATA_PATH = FILE_PATH / 'data'
MODELS_PATH = FILE_PATH / 'models'
UTILS_PATH = FILE_PATH / 'utils'
sys.path.append(str(UTILS_PATH))

import kmeans_tfid as tm 

def get_kmean_model(party, clusters, iter = 100):
    print(f'Getting kmeans model of party{party}')
    
    k = clusters[party]
    tweets = pd.read_csv(DATA_PATH / f'tweets_party{party}_cleaned.csv', dtype=str)
    data = tweets['full_text_cleaned']
    
    print('Fitting the model')
    x, vect = tm.obtain_tfidf_terms(data)
    kmeans,labels = tm.obtain_kmeans_model(x, k, iter)
    
    print('Topics obtained:')
    tm.get_top_keywords(kmeans, vect.get_feature_names(), k, 10)
    
    print('Saving the terms transformated, the model and labels')
    with open(MODELS_PATH / f'x_kmeanstfidf_k{k}_party{party}.pkl', 'wb') as file:
        pickle.dump(x, file)
    with open(MODELS_PATH / f'model_kmeanstfidf_k{k}_party{party}.pkl', 'wb') as file:
        pickle.dump(kmeans, file)
    with open(MODELS_PATH / f'labels_kmeanstfidf_k{k}_party{party}.pkl', 'wb') as file:
        pickle.dump(labels, file)
    with open(MODELS_PATH / f'vect_kmeanstfidf_k{k}_party{party}.pkl', 'wb') as file:
        pickle.dump(vect, file)


clusters={1:16,2:16,3:10,4:12, 5:16}
iter=500
for party in clusters.keys():
    get_kmean_model(party, clusters,iter)
