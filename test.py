import tweepy
import json
from datetime import datetime
import langdetect
import numpy as np
from copy import copy
import xlsxwriter
import logging 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Collecting users from Twitter')
# Get the User object from a user account


class Twitter:
    def __init__(self):
        """
        locationname : str 
        granularity : str  default 'country'
        tweets_seed : int
                        Number of tweets to obtain the initial group of users
        min_n_tweets : int 
                       Minimum number of tweets to be considered useful
        min_n_friends : int
                        Minimum number of friends to be considered useful
        tweet_language : str 
                        Expected language of the tweets. Default 'it'
        n_users: int
                 Total number of unique users to obtain
        sample_size : int 
                       Number of users to choose from the user list to
                       walk trough their followers
        followers_sample_size : int
                                How many followers get from each sampled user
        n_tweets_per_user : int 
                            Indicates how many tweets per user to save
        """

        self.locationname = 'Italy'
        self.granularity = 'country'
        self.tweets_seed = 100
        self.min_n_tweets = 10
        self.min_n_friends = 15
        self.tweet_language = 'it'
        self.n_users = 1000
        self.sample_size = 5
        self.friends_sample_size = 100
        self.n_tweets_per_user = 15
        self.last_tweets_date = "2019-10-20"
        self.min_date_activity_user = datetime(2019, 6, 1, 0, 0)
        self.max_date_activity_user = datetime(2019, 9, 30, 0, 0)
        self.get_tweets = False
        self.init_api()

    def init_api(self):
        # Load credentials from json file
        with open("twitter_emi_credentials.json", "r") as file:
            credentials = json.load(file)

        consumer_key = credentials['CONSUMER_KEY']
        consumer_secret = credentials['CONSUMER_SECRET']
        access_token = credentials['ACCESS_TOKEN']
        access_token_secret = credentials['ACCESS_SECRET']

        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)

        self.api = tweepy.API(auth,
                              wait_on_rate_limit=True,
                              wait_on_rate_limit_notify=True)

    def user_useful(self, userid):
        """
            Return wether the user is useful or not

            Arguments
            -----------
             
            userid: The user id to check
        """
        user_info = self.api.get_user(userid)
        if (user_info.statuses_count < self.min_n_tweets):
            return False
        if (user_info.friends_count < self.min_n_friends):
            return False

        try:
            status = self.api.user_timeline(id=userid, count=1)
            last_status = status[0]
        except tweepy.error.TweepError:
            return False

        
        # the user had activity in the last 'min_date_last_activity_user' days
        if (last_status.created_at < self.min_date_activity_user):
            return False
        
        # 'the lang is the same that it country'
        try:
            if (langdetect.detect(last_status.text) != self.tweet_language):
                return False
        except langdetect.lang_detect_exception.LangDetectException:
            return False
        return True
        


    def get_tweets_by_location(self):
        list_users = []
        places = self.api.geo_search(query=self.locationname,
                                     granularity=self.granularity)
        place = places[0]
        public_tweets = tweepy.Cursor(self.api.search,
                                      q="place:%s" % place.id,
                                      since=self.last_tweets_date,
                                      show_user=True,
                                      tweet_mode="extended").items(self.tweets_seed)
        tweets = list(public_tweets)
        for tweet in tweets:
            user_id = tweet.user.id
            if user_id not in list_users:
                if (self.user_useful(user_id)):
                    list_users.append(user_id)
        return list_users

    def get_users_for_analysis(self, initial_users):
        list_users = copy(initial_users)
        while (len(list_users) < self.n_users):
            logger.info(f'Number of users in the final list: {len(list_users)}')
            sampling = np.random.choice(list_users, 
                                        size=min(self.sample_size, len(list_users)), 
                                        replace=False)
            logger.info(f'Number of users choosed: {len(sampling)}')
            logger.info(f'Procesing friends')
            for user in sampling:
                friends_ids = self.api.friends_ids(user_id=user)
                if len(friends_ids) > 0:
                    friends_ids = list(set(friends_ids) - set(list_users))
                    friends_ids = np.random.choice(friends_ids,
                                                   size=min(self.friends_sample_size, len(friends_ids)), 
                                                   replace=False)
                    for id in friends_ids:
                        if self.user_useful(id):
                            list_users.append(id)
                            logger.info(f'Number of users in the final list: {len(list_users)}')
                
        return list_users


    def obtain_users(self):
        workbook = xlsxwriter.Workbook('users.xlsx')
        users_worksheet = workbook.add_worksheet('Users')
        users_worksheet.set_column(0, 0, 25)
        users_worksheet.set_column(1, 1, 15)
        users_worksheet.set_column(2, 2, 30)
        users_worksheet.set_column(3, 3, 15)
        tweets_worksheet = workbook.add_worksheet('Tweets')
        logger.info('Obtaining the initial seed of users')
        first_users = self.get_tweets_by_location()
        logger.info(f'Number of users in the initial list: {len(first_users)}')
        logger.info('Obtaining the complete set of users')
        users = self.get_users_for_analysis(first_users)
        logger.info('Ready the list of users :)')
        row = 0
        for i, user_id in enumerate(users):
            logger.info(f'{i} of {len(users)}')
            user = self.api.get_user(user_id)
            users_worksheet.write(i, 0, str(user_id))
            users_worksheet.write(i, 1, user.screen_name)
            url_user = 'https://twitter.com/' + user.screen_name
            users_worksheet.write_url(i, 2, url_user)
            if self.get_tweets :
                logger.info('Obtaining the tweets of selected users')
                users_worksheet.write(i, 3, f'internal:Tweets!B{row+1}')                                
                tweets_worksheet.write(row, 0, user.screen_name)
                self.write_tweets(user_id, tweets_worksheet, row)
                
        workbook.close()    

    def write_tweets(self, user_id, tweets_worksheet, row):
        tweets = tweepy.Cursor(self.api.user_timeline, id=user_id).items()
        tweets_added = 0
        for tweet in tweets:
            if tweet.created_at < self.min_date_activity_user:
                break
            if tweets_added == self.n_tweets_per_user:
                break
            if tweet.created_at <= self.max_date_activity_user:
                if tweet.created_at >= self.min_date_activity_user: 
                    tweets_worksheet.write(row, 1, str(tweet.created_at))
                    tweets_worksheet.write(row, 2, tweet.text)
                    row += 1

def get_user_information(api, username):
    user = api.get_user(username)
    print(user.screen_name)
    print(user.id)
    print(user.followers_count)
    print(user.friends_count)  # accounts that the user follows
    print(user.description)
    print(user.location)
    print(user.statuses_count)  # number of tweets of the user
    # print(user)

a = Twitter()
a.obtain_users()