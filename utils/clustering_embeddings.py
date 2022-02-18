import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from fastai.text import load_learner
from tqdm import tqdm

from pathlib import Path
#
# Configure the path
ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / 'data'
MODELS_PATH = ROOT_PATH / 'models'
UTILS_PATH = ROOT_PATH / 'utils'

import logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('clustering')


def load_classifier(classifier_exported):
        return load_learner(MODELS_PATH,classifier_exported)


def get_embedding(question, clf):
        tensor = list(clf.model.children())[0].forward(
                clf.data.one_item(question)[0])[0][-1][-1][-1]
        return tensor.detach().numpy()


def get_save_embeddings(classifier_exported, data, text_column, embeddings_filename):
    logger.info('Loading the classifier')
    text_clas = load_classifier(classifier_exported)
    logger.info('Getting the embeddings')
    embeddings = []
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        embeddings.append(get_embedding(row[text_column], text_clas))
    data['embeddings'] = embeddings

    embeddings_filename = DATA_PATH / embeddings_filename
    logger.info(f'Saving the embeddings in file: {embeddings_filename}')
    data.to_pickle(embeddings_filename)
    print(data[[text_column, 'embeddings']].head())


def get_clusters(umlfit_embeddings, kmeans_embeddings_results_filename):
    logger.info('Loading the data')
    df = pd.read_pickle(DATA_PATH / umlfit_embeddings)
    X = df['embeddings'].to_list()
    kmeans = KMeans(n_clusters=4)
    logger.info('Obtaining the KMeans model')
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    df['cluster'] = y_kmeans
    OUTPUT_FILE = DATA_PATH / kmeans_embeddings_results_filename
    logger.info(f'Saving the data with the predicted topics: {OUTPUT_FILE}')
    df.to_pickle(OUTPUT_FILE)



