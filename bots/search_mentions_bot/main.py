import tweepy
import logging
import time
import os
from os import getenv
from google.cloud import storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Authenticate to Twitter
def create_api():
    api_key = getenv("api_key")
    api_key_secret = getenv("api_key_secret")
    access_token = getenv("access_token")
    access_token_secret = getenv("access_token_secret") 

    auth = tweepy.OAuth1UserHandler(api_key, api_key_secret, access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    try:
        api.verify_credentials()
    except Exception as e:
        logger.error("Error creating API", exc_info=True)
        raise e
    
    logger.info("API created successfully!")
    
    return tweepy.API(auth)
    
# Searches mention from tweets and responds using cursor
# it gets only one previous tweet from the caller. Makes it easier for testing.
def searches_mentions(api, keywords, since_id):
    logger.info("Retrieving mentions")
    new_since_id = since_id
    for tweet in tweepy.Cursor(api.mentions_timeline, since_id=since_id).items():
        new_since_id = max(tweet.id, new_since_id)
        print(tweet.id)
        if tweet.in_reply_to_status_id is None:
            continue
        if any(keyword in tweet.text.lower() for keyword in keywords):
            logger.info('Answering to %s', tweet.user.name)
            try:
                original_tweet_id = tweet.in_reply_to_status_id
                original_tweet = api.get_status(original_tweet_id)
                logger.info("tweet: %s", original_tweet.text)
                
                response = "got info of tweet: " + original_tweet.text
                api.update_status(status=response, in_reply_to_status_id=tweet.id,  auto_populate_reply_metadata=True)

            except tweepy.errors.NotFound as e:
                    logging.error("User or tweet deleted: {}".format(e))

            except tweepy.errors.TweepyException as e:  
                logging.error("Tweepy error occured:{}".format(e))
    return new_since_id

def get_since_id(bucket, filename):
    last_tweet_id = bucket.blob("last_tweet_id.txt").download_as_text()
    return last_tweet_id

def update_file(new_since_id, bucket, filename):
    blob = bucket.blob(filename)
    blob.upload_from_string(new_since_id)

def main(request):
    api = create_api()
    # bot_id = api.verify_credentials().id

    storage_client = storage.Client()
    bucket = storage_client.bucket("tcc-tweet-bot")

    since_id = int(get_since_id(bucket, "last_tweet_id.txt"))
    since_id = searches_mentions(api, ["help", "support", "calling"], since_id)
    
    update_file(str(since_id), bucket, "last_tweet_id.txt")
    return str(since_id)