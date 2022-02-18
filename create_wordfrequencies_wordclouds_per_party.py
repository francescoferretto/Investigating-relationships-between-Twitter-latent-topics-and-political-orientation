import pandas as pd 
import itertools
import os
import numpy as np
import text_processing as tp
import topic_analysis as ta
import pandas as pd
from pathlib import Path
#
# Configure the path
FILE_PATH = Path(__file__).resolve().parent
DATA_PATH = FILE_PATH / 'data'
FIGURES_PATH = FILE_PATH / 'figures'
os.chdir(str(FILE_PATH))

for party in np.arange(1,6):

    tweets = pd.read_csv(DATA_PATH / f'tweets_party{party}_cleaned.csv')
    texts = tweets['full_text_cleaned']
    texts = texts.str.split()
    clean_words = list(itertools.chain(*texts))

    # Obtaining most common words
    most_common_words=ta.calculate_words_frecuencies(clean_words, n=100)
    fig = ta.plot_words_frecuencies(most_common_words[0:20])
    file = FIGURES_PATH / f'mostcommon_tweets_party{party}.png'
    fig.savefig(file)

    # Obtaining word cloud
    fig = ta.get_word_cloud(most_common_words)
    file = FIGURES_PATH / f'wordcloud_tweets_party{party}.png'
    fig.to_image()
    fig.to_file(file)

    