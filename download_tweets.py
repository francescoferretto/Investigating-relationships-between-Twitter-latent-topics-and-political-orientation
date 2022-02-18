import tweepy 
import csv 
import sys
import json
import pandas as pd
import os
import logging 

# Configure the path 
os.chdir('/home/emi/unipd/Sartori_CBSD/project/cbsdproject')

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Downloading tweets')

reload(sys)  
sys.setdefaultencoding('utf-8')

# Load credentials from json file
with open("twitter_emi_credentials.json", "r") as file:
	credentials = json.load(file)

CONSUMER_KEY = credentials['CONSUMER_KEY']
CONSUMER_SECRET = credentials['CONSUMER_SECRET']
ACCESS_TOKEN = credentials['ACCESS_TOKEN']
ACCESS_SECRET = credentials['ACCESS_SECRET']

# Load the Recording File to get the users
data = pd.read_excel('RecordingFile_GroupC_task3_deliver.xlsx', skiprows=8)
users = data['User Name'].tolist()
first_row = 0
last_row = 200
handles_list = users[first_row:last_row]
#handles_list = ["SrBachchan", "imVkohli", "sachin_rt"]

# define a variable to save the users that deleted their account
failed_accounts = []


def get_all_tweets(screen_name):

	auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
	auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
	api = tweepy.API(auth)	
	
	alltweets = []
	new_tweets = api.user_timeline(screen_name = screen_name,count=200,tweet_mode='extended')
	
	
	alltweets.extend(new_tweets)
 	oldest = alltweets[-1].id - 1
        
				
	while len(new_tweets) > 0:		
		logger.info("getting tweets before %s" % (oldest))
		new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest,tweet_mode='extended')
		alltweets.extend(new_tweets)
		oldest = alltweets[-1].id - 1
		logger.info("...%s tweets downloaded so far" % (len(alltweets)))

	outtweets = [[tweet.id_str,
				tweet.created_at, 
				tweet.favorite_count,
				tweet.retweet_count,
				tweet.source.encode("utf-8"),
				tweet.full_text.encode("utf-8")] for tweet in alltweets]
					
	logger.info('Writing the ouput in a file')
	with open('tweets/%s_tweets.csv' % handle, 'wb') as f:
		writer = csv.writer(f)
		writer.writerow(["id",
				"created_at",
				"favorites",
				"retweets",
				"source",
				"full_text"])
		writer.writerows(outtweets)
	pass


if __name__ == '__main__':     
		for handle in handles_list:
			if os.path.exists('tweets/%s_tweets.csv' % (handle)):
				continue
			logger.info("getting tweets of the user %d: %s" % (handles_list.index(handle), handle))
			try:
				get_all_tweets(handle)
			except tweepy.error.TweepError:
				failed_accounts.append(handle)
		logger.info('Number of downloaded fail accounts: %s' % (len(failed_accounts)))
		if (len(failed_accounts) > 0):
			logger.info('Account/s deleted/suspended/failed.. is/are: {}'.format(", ".join(failed_accounts)))
		logger.info("Done.")