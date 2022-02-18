import pandas as pd
import numpy as np
import sys
from pathlib import Path
#
# Configure the path
FILE_PATH = Path(__file__).resolve().parent
DATA_PATH = FILE_PATH / 'data'
FIGURES_PATH = FILE_PATH / 'figures'
UTILS_PATH = FILE_PATH / 'utils'
sys.path.append(str(UTILS_PATH))

import kmeans_tfid as tm 

for party in np.arange(1,6):
    tweets = pd.read_csv(DATA_PATH / f'tweets_party{party}_cleaned.csv')
    data = tweets['full_text_cleaned'].dropna()

    max_clusters = 20
    iterations = 200
    print(f'looking for the best number of clusters of party: {party}')
    x, vect = tm.obtain_tfidf_terms(data)

    sse = tm.find_optimal_clusters(x, max_clusters, iterations)
    print('plotting')
    file = FIGURES_PATH / f'best_clusters_kmeanstfidf_party{party}.png'
    tm.plot_sse(sse, max_clusters, file)