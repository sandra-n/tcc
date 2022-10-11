import tweepy
import logging
import twitter_keys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Authenticate to Twitter
def create_api():
    auth = tweepy.OAuthHandler(twitter_keys.api_Key, twitter_keys.api_key_secret)
    auth.set_access_token(twitter_keys.access_token, twitter_keys.access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    try:
        api.verify_credentials()
    except Exception as e:
        logger.error("Error creating API", exc_info=True)
        raise e
    
    logger.info("API created successfully!")
    
    return tweepy.API(auth)

def reply_tweet(tweet_id, api):
    tweet = api.get_status(tweet_id)
    users_mentioned_array = tweet.entities.get('user_mentions')
    users_mentioned = [users['screen_name'] for users in users_mentioned_array if users['screen_name'] == api.verify_credentials().screen_name]
    if users_mentioned:
        original_tweet_id = tweet.in_reply_to_status_id
        original_tweet = api.get_status(original_tweet_id)
        logger.info("tweet: %s", original_tweet.text)
    else:
        logger.info("I was not called here!")

#searches mention for username and gets original tweet
class SearchesMentionAndPrintOriginalTweet(tweepy.Stream):
    def __init__(self, *args, api):
        super().__init__(*args)
        self.api = api
        self.me = api.verify_credentials()

    def on_status(self, tweet):
        logger.info("Processing tweet id %s", tweet.id)
        if tweet.in_reply_to_status_id is not None and tweet.in_reply_to_user_id != self.me.id:
            logger.info("Tweet entities: %s", tweet.entities)
            users_mentioned_array = tweet.entities.get('user_mentions')
            users_mentioned = [users['screen_name'] for users in users_mentioned_array if users['screen_name'] == self.me.screen_name]
            if users_mentioned:
                original_tweet_id = tweet.in_reply_to_status_id
                original_tweet = self.api.get_status(original_tweet_id)
                logger.info("tweet: %s", original_tweet.text)
            else:
                logger.info("I was not called here!")

    def on_error(self, status):
        logger.error(status)

def main(keywords):
    #mention_searcher = SearchesMentionAndPrintOriginalTweet(twitter_keys.bearer_token)
    #mention_searcher.sample()
    user_api = create_api()
    #reply_tweet('1578115883922296832', user_api)
    tweets_listener = SearchesMentionAndPrintOriginalTweet(
        twitter_keys.api_Key, 
        twitter_keys.api_key_secret, 
        twitter_keys.access_token, 
        twitter_keys.access_token_secret, api=user_api)
    tweets_listener.filter(track=keywords, languages=["en"])

if __name__ == "__main__":
    main(["@gabrielsanefuji"])

