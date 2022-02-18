import pandas as pd
import text_processing as tp
import topic_analysis as ta
import itertools
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import matplotlib.cm as cm
from sklearn.decomposition import TruncatedSVD


def obtain_tfidf_terms(data):
    vect = TfidfVectorizer()
    x = vect.fit_transform(data)
    return x, vect


def obtain_kmeans_model(x, k=4, iter=100):
    kmeans = MiniBatchKMeans(n_clusters=k,
                             init='k-means++',
                             max_iter=iter,
                             batch_size=int(x.shape[0] / 2))
    labels = kmeans.fit_predict(x)
    return kmeans, labels


def get_top_keywords(model, terms, clusters, n_terms=10):
    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    for i in range(clusters):
        print("Cluster %d:" % i, )
        topic = []
        for ind in order_centroids[i, :n_terms]:
            topic.append(terms[ind])
        print(', '.join(topic))


def create_tsne_pca(data, max_items):
    print('getting pca')
    pca = TruncatedSVD(n_components=50).fit_transform(data[max_items, :])
    print('getting tsne')
    tsne = TSNE().fit_transform(pca)
    return tsne, pca


def plot_tsne_pca(data, labels, file):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]),
                                 size=3000,
                                 replace=False)
    print(f'max items: {max_items}')
    print('getting pca and tsne transformation')
    tsne, pca = create_tsne_pca(data, max_items)

    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    label_subset = [cm.hsv(i / max_label) for i in label_subset[idx]]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].scatter(pca[idx, 0], pca[idx, 1], c=label_subset)
    ax[0].set_title('PCA Cluster Plot')

    ax[1].scatter(tsne[idx, 0], tsne[idx, 1], c=label_subset)
    ax[1].set_title('TSNE Cluster Plot')
    fig.savefig(file)


def find_optimal_clusters(data, max_k, iter=100):
    iters = range(2, max_k + 1, 2)
    sse = []
    for k in iters:
        sse.append(
            MiniBatchKMeans(n_clusters=k,
                            init='k-means++',
                            max_iter=iter,
                            init_size=1024,
                            batch_size=2048,
                            random_state=20).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
    return sse


def plot_sse(sse, max_k, file):
    iters = range(2, max_k + 1, 2)
    fig, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    fig.savefig(file)
