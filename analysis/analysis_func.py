import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.retrieval_func import pull_tweets, tag_activity
from processing.tweet_processing import tweet_processing
from twarc import expansions
from math import log1p, sqrt

def min_max_normalize(val, min_val, max_val):
    """
    Normalize a value to a range of 0 to 1 based on the provided min and max values.
    Parameters:
    val (float): The value to normalize.
    min_val (float): The minimum value of the range.
    max_val (float): The maximum value of the range.
    Returns:
    float: The normalized value between 0 and 1.
    If max_val equals min_val, returns 0.0 to avoid division by zero.
    """
    if max_val - min_val == 0:
        return 0.0
    return (val - min_val) / (max_val - min_val)

def discourse_temp(tweets, activity_dict):
    '''
    Calculate the discourse temperature based on tweet data and activity.
    Parameters:
    tweets (list): A list of processed tweet dictionaries.
    activity_dict (dict): A dictionary containing activity data with 'counts' and 'times'.
    Returns:
    float: The discourse temperature value.
    The temperature is calculated as the sum of the mean absolute sentiment, mean activity, and a scaled engagement score.
    The engagement score is a combination of average like, retweet, reply, and quote counts.
    The follower count is also scaled using log1p to avoid skewing the results with large values.
    The final temperature is normalized to a range of 0 to 150 and a value between 0 and 1 is returned.
    '''
    
    follower_counts = np.array([tweet['follower_count'] for tweet in tweets if 'follower_count' in tweet])
    avg_follower_count = np.mean(follower_counts) if follower_counts.size > 0 else 0

    sent_counts = np.array([abs(tweet['sentiment']) for tweet in tweets if 'sentiment' in tweet])
    mean_abs_sent = np.mean(sent_counts) if sent_counts.size > 0 else 0
    
    mean_activity = np.mean(activity_dict['counts'])

    like_counts = [tweet['public_metrics']['like_count'] for tweet in tweets if 'public_metrics' in tweet and 'like_count' in tweet['public_metrics']]
    retweet_counts = [tweet['public_metrics']['retweet_count'] for tweet in tweets if 'public_metrics' in tweet and 'retweet_count' in tweet['public_metrics']]
    reply_counts = [tweet['public_metrics']['reply_count'] for tweet in tweets if 'public_metrics' in tweet and 'reply_count' in tweet['public_metrics']]
    quote_counts = [tweet['public_metrics']['quote_count'] for tweet in tweets if 'public_metrics' in tweet and 'quote_count' in tweet['public_metrics']]
    avg_like_count = np.mean(like_counts) if like_counts else 0
    avg_retweet_count = np.mean(retweet_counts) if retweet_counts else 0
    avg_reply_count = np.mean(reply_counts) if reply_counts else 0
    avg_quote_count = np.mean(quote_counts) if quote_counts else 0
    
    engagement_score = avg_like_count + avg_retweet_count**2 + avg_reply_count**2 + avg_quote_count**2

    scaled_follower = log1p(avg_follower_count)
    scaled_engagement = log1p(engagement_score)

    temp = mean_abs_sent + mean_activity + (scaled_follower * scaled_engagement)
    
    scaled_temp = min_max_normalize(temp, 0, 150)

    return scaled_temp

def main():
    tags = ["#BlackLivesMatter", "TSTheLifeofaShowgirl",]
    for tag in tags:
        query = f"{tag} lang:en -is:retweet"
        activity_dict = tag_activity(query)
        search_results = pull_tweets(query, max_results=10)
        first_page = next(search_results)
        # The Twitter API v2 returns the Tweet information and the user, media etc. separately
        # so we use expansions.flatten to get all the information in a single JSON
        result = expansions.flatten(first_page)
        processed_tweets = tweet_processing(result)
        
        temp = discourse_temp(processed_tweets, activity_dict)
        print(f"Discourse Temperature of {tag}: {temp}")


if __name__ == "__main__":
    main()