import gzip
import logging
import pickle
import re
import time

import matplotlib.pyplot as plt
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
import text_processing as ca
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel, LdaModel
from wordcloud import WordCloud
import seaborn as sns
import numpy as np
import circlify as circ
import matplotlib
from skimage import draw

def plot_words_frecuencies(most_common_words, figsize=(10,14), fontsize_labels=14):
    sns.set()
    plt.figure(figsize=figsize)
    plt.rc('xtick', labelsize=fontsize_labels)    # fontsize of the x and y labels
    plt.rc('ytick', labelsize=fontsize_labels)
    colors = sns.light_palette((260, 75, 60), input="husl", n_colors=40, reverse=True)
    fig = sns.barplot(x=[val[1] for val in most_common_words], y=[val[0] for val in most_common_words], palette=colors)
    return fig

#get n-words frecuencies
def calculate_words_frecuencies(lists_words, n=30):
    """Calcute the most n frequents words

    Arguments:
        lists_words: list of str
              List of words

    Keyword Arguments:
        n: number of most frequents words
                   default=30
    Returns:
        list of 
    """
    word_freqdist = nltk.FreqDist(w for w in lists_words)
    most_common = word_freqdist.most_common(n)
    return most_common

#get bigram frecuencies
def calculate_bigram_frecuencies(lists_words):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    words = ca.cleaning_tokenize(lists_words)
    finder = BigramCollocationFinder.from_words(words)
    finder.nbest(bigram_measures.pmi, 5)
    return finder


def add_other_stopwords(stop_words, other_stop_words):
    stop_words.update(other_stop_words)
    return stop_words


def get_word_cloud(words, stop_words=[], collocations=True, color='white'):
    """Generate the word cloud with the most n frequents words

    Arguments:
        words: list of tupls or str
              List of words with their frequencies (we can obtain with the method calculate_words_frecuencies) 
              or a String with all the text

        stop_words: list of str
            List of stopwords
        collocations: Boolean
        color: str
    Returns:
        wordcloud 
    """
    logging.info('Generating word cloud')
    wordcloud = WordCloud(
        background_color=color,
        stopwords=stop_words,
        max_words=200,
        max_font_size=40,
        collocations=collocations,
        )
    if isinstance(words, str):
        wordcloud.generate(words)
    elif isinstance(words, list):
        words = {a[0]:a[1] for a in words}
        wordcloud.generate_from_frequencies(words)
    return wordcloud


def get_best_kmeans_k_parameter(data, clusters, iter):
    vect = TfidfVectorizer()
    x = vect.fit_transform(data.values)
    sse = {}
    for k in range(1, clusters):
        logging.info("Number of topics:" + str(k))
        kmeans = MiniBatchKMeans(n_clusters=k,
                init='k-means++',
                max_iter=iter,
                batch_size=int(x.shape[0]/20)).fit(x)
        sse[k] = kmeans.inertia_  # Inertia: Sum of squared distances of samples to their closest cluster center.
    return sse


def compute_coherence_values(dictionary, corpus, texts, clusters, iter=100):
    coherence_values = []
    model_list = []
    for num_topics in range(2, clusters):
        logging.info("Number of topics:" + str(num_topics))
        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, iterations=iter)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def get_best_k_parameter_lda(data, filename, clusters, iter):
    # create a Gensim dictionary from the texts
    dictionary = corpora.Dictionary(data)
    # remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
    dictionary.filter_extremes(no_below=1, no_above=0.8)
    # convert the dictionary to a bag of words corpus for reference
    corpus = [dictionary.doc2bow(text) for text in data]
    # Can take a long time to run.
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=data, clusters=clusters, iter=iter)
    # Show graph
    x = range(2, clusters)
    return x, coherence_values


def topic_cloud(topics):
    def generate_word_cloud(words, mask):
        wordcloud = WordCloud(background_color=None,
                              max_words=200,
                              width=1200,
                              height=1200,
                              max_font_size=90,
                              mode='RGBA',
                              colormap='Set1',
                              color_func=lambda *args, **kwargs: (0, 0, 0),
                              mask=mask)

        words = {a[0]: a[1] for a in words}
        wordcloud.generate_from_frequencies(words)
        return wordcloud

    def generate_mask():
        arr = np.ones((200, 200, 3)) * 255
        rr, cc = draw.circle(100, 100, radius=100, shape=arr.shape)
        arr[rr, cc, :] = [0, 0, 0]
        return arr

    mask = generate_mask()
    circles = circ.circlify([d[0] for d in topics],
                            show_enclosure=False,
                            target_enclosure=circ.Circle(1, 1, 1))
    clouds = [generate_word_cloud(d[1], mask) for d in topics]

    fig, ax = plt.subplots(1, 1, figsize=(25, 15))

    cmap = sns.color_palette("Paired", len(circles))
    for i, (circle, cloud) in enumerate(zip(circles, clouds)):
        nrows, ncols = ax.get_subplotspec().get_gridspec().get_geometry()
        extent = (circle.x - ncols * circle.r, circle.x + ncols * circle.r,
                  circle.y - nrows * circle.r, circle.y + nrows * circle.r)

        ax.imshow(cloud.to_image(), interpolation='lanczos', extent=extent)
        back_color = np.array([*cmap[i], 0])
        back_color[3] = 0.2
        circle1 = plt.Circle((circle.x, circle.y),
                             circle.r,
                             color=back_color,
                             zorder=0)
        ax.add_artist(circle1)
    ax.axis('off')
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    return fig