## Investigating relationships between Twitter latent topics and political orientation

Project for the Cognitive, Behavioral and Social Data @ Unipd, made by Natascia Caria, Marìa Emilia Charnelli, Francesco Ferretto, Matteo Lavina and Andrea Sinigaglia

The aim of this project was, after a preliminary phase of data collection and labeling, in finding topics by analyzing the cluster of topics among different political parties and see their evolution through time. 
For an in-depth description of the project with its results, look at "CBSD_project.pdf" file.



### Install Packages / Libraries that the project uses
* tweepy
* sklearn
* numpy
* nltk
* progressbar2
* gensim
* wordcloud
* many_stop_words 
* stop_words
* xlrd 
* scikit-image
* circlify
* fastai
* tqdm
* python-box


### Create folders:
* figures -> folder to save the figures
* models -> folder to save the NLP models created
* data -> to save the temporal data generated like tweets per party, filter accounts
* Tweets -> create the folder Tweets with all the tweets 

### Credentials

* You need to create a json file like this:

```
{"CONSUMER_KEY": "jjj", 
"CONSUMER_SECRET": "eeeRR", 
"ACCESS_TOKEN": "aaaax", 
"ACCESS_SECRET": "asadd"}
```

Or you can use the script "create_credentials_json.py" to create a json file with the credentials

### Files


* filter_valid_accounts.py -> Script to clean the file data of the twitter accounts and filter the valid accounts (users that have the tweets files)

* create_parties_tweets_files.py -> Script to clean tweets per party using the valid accounts obtained in the previous script and save the tweets per party into csv files in the folder data

* obtain_best_k_parameter_kmeanstfidf_per_party.py -> Script to obtain the best number of cluster per party for the Kmeans TFIDF algorithm

* obtain_kmeans_model_per_party.py -> Script to obtain the model of topics per party using the number of clusters obtained in the previous script for each party.

* create_file_allparties_tweets.py -> script to create a file with all the tweets of all the parties

* obtain_umlfit_party_classifier.py -> Script to obtain a ULMFit party classifier using Fastai

## Folder utils:

* create_credentials_json.py -> Script to create the json file with the credentials of a Twitter account.

* text_processing.py -> utils functions used in create_parties_tweets_files.py scripts to clean tweets content.

* topic_analysis.py -> utils functions used for the topics modeling

* kmeans_tfidf.py -> utils functions to use KMeans algorithm for the topics modeling

* google-10000-english.txt -> Dictionary of English Words created per Google using in tex_processing to clean the tweets.

* ulmfit_fastai.py -> Utils functions for using Fastai

#### Folder Notebooks

* data_exploration.ipynb -> First notebook to explore the accounts labeled file
* data_analysis.ipynb -> Second notebook to analize the valid accounts and how many tweets we have
* most_common_words_parties.ipynb -> Third notebook with the most frequencies words per party
* topics_topwords_parties_KMeansTFIDF.ipynb -> Fourth notebook with the topics per party obtained using KMeans TFIDF
* topics_temporality_parties_kmeans_tfidf.ipynb -> Plot the frequencies around the time of the topics per party

