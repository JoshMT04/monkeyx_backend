import json
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.retrieval_func import pull_tweets
from twarc import expansions
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize

analyser = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    '''
    Preprocess the text by removing URLs, punctuation, and stop words.
    Parameters:
    text (str): The text to preprocess.
    Returns:
    list: A list of filtered tokens.
    '''

    # Removing URLS
    text = re.sub(r"https?://\S+|www\.\S+"," ",text)
    
    #Removing punctuation and special caracteres 
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    
    #Treating stop words
    filtered_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 3:
            filtered_tokens.append(token)
    
    return filtered_tokens

def get_sentiment(text):
    '''
    Get the sentiment score of the text using VADER.
    Parameters:
    text (str): The text to analyze.
    Returns:
    float: The sentiment score of the text. Ranges from -1 (negative) to 1 (positive).
    '''
    if not text or not isinstance(text, str):
        return None
    return analyser.polarity_scores(text)['compound']

def tweet_processing(tweets):
    '''
    Process tweets to extract relevant information and store in a dictionary.
    Parameters:
    tweets (list): A list of tweet dictionaries to process.
    Returns:
    list: A list of processed tweets with relevant information.
    '''
    # First, extract referenced tweets (if they exist) to be treated as an independent tweet
    # The relationships are stored in the 'referenced_tweets' field to link to the original tweet
    # Boolean markers of 'is_reference' and 'is_original' are used to distinguish
    og_tweets = tweets.copy()
    for tweet in og_tweets:
        if 'referenced_tweets' in tweet:
            referenced_tweet = tweet['referenced_tweets'][0].copy()
            referenced_tweet['is_reference'] = True
            referenced_tweet['reference_id'] = tweet['id']
            tweets.append(referenced_tweet)
            
        tweet['is_reference'] = False
        
    # Second, retrieve the author information from the 'author_id' field
    for tweet in tweets:
        if 'author_id' in tweet:
            author_username = tweet['author']['username']
            follower_count = tweet['author']['public_metrics']['followers_count']
            tweet['author'] = author_username
            tweet['follower_count'] = follower_count
            
    # Third, perform binary sentiment analysis on the tweet text
    # This first required preprocessing the text to remove URLs, punctuation, and stop words
    # VADER from NLTK is then used for sentiment analysis
    for tweet in tweets:
        if 'text' in tweet:
            filtered_tokens = preprocess_text(tweet['text'])
            sentiment = get_sentiment(" ".join(filtered_tokens))
            tweet['sentiment'] = sentiment if sentiment is not None else 0
        else:
            tweet['sentiment'] = 0
            tweet['text'] = ""
    
    for tweet in tweets:
        if 'public_metrics' in tweet:
            tweet['like_count'] = tweet['public_metrics']['like_count']
            tweet['retweet_count'] = tweet['public_metrics']['retweet_count']
            tweet['reply_count'] = tweet['public_metrics']['reply_count']
            tweet['quote_count'] = tweet['public_metrics']['quote_count']

    keep_keys = ['id', 'text', 'like_count', 'retweet_count', 'reply_count', 'quote_count', 'created_at', 'author',
                 'is_reference', 'reference_id', 'follower_count', 'sentiment']
    filtered_tweets = []
    for tweet in tweets:
        filtered_tweet = {k: tweet[k] for k in keep_keys if k in tweet}
        filtered_tweets.append(filtered_tweet)
        
    return filtered_tweets

def main():
    query = "#FreePalestine lang:en -is:retweet"
    search_results = pull_tweets(query, max_results=10)
    first_page = next(search_results)
    # The Twitter API v2 returns the Tweet information and the user, media etc. separately
    # so we use expansions.flatten to get all the information in a single JSON
    result = expansions.flatten(first_page)
    print(result[0])
    processed_tweets = tweet_processing(result)
    df = pd.DataFrame(processed_tweets)
    df.to_csv('tweet_log/140825_palestine_processed.csv', index=False)

if __name__ == "__main__":
    main()