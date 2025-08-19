# This will import the Twarc2 client and expansions class from twarc library and also the json library
from twarc import Twarc2, expansions
import os
from dotenv import load_dotenv
import json

# This is where you initialize the client with your own bearer token
load_dotenv()
bearer_token = os.getenv('X_API_BEARER')
client = Twarc2(bearer_token=bearer_token)

def tag_activity(query):
    """
    This function retrieves the recent Tweet counts for a specific hashtag.
        
    Parameters:
    query (str): The query string to filter tweets, e.g., '#FreePalestine lang:en -is:retweet'.
    Returns:
    dict: A dictionary containing the times and counts of tweets for the specified query.
    The dictionary has two keys:
        - 'times': A list of timestamps when the tweets were counted.
        - 'counts': A list of tweet counts corresponding to the timestamps.
    Example:
    >>> tag_activity("#FreePalestine lang:en -is:retweet")
    {
        'times': ['2023-10-01T00:00:00Z', '2023-10-01T01:00:00Z', ...],
        'counts': [10, 15, ...]
    } 
    """
    # Get the tweet counts for the specified query
    # Granularity is set to "hour" to get hourly counts
    count_results = client.counts_recent(query=query, granularity="hour")
    
    # Initialise lists to store times and counts
    times, counts = [], []
    
    # Iterate through the results to extract time and tweet counts
    for page in count_results:
        for bucket in page['data']:
            times.append(bucket['start'])
            counts.append(bucket['tweet_count'])

    return {'times': times, 'counts': counts}

def pull_tweets(query, max_results=10):
    '''
    Function to pull tweets based on a query.
    Parameters:
    query (str): The query string to filter tweets, e.g., '#FreePalestine lang:en -is:retweet'.
    max_results (int): The maximum number of tweets to return per page. Default is 10.
    Returns:
    search_results: A twarc generator that yields pages of tweets matching the query.
    
    '''
    # The search_recent method calls the recent search endpoint to get Tweets based on the query, start and end times
    # Tweets are pulled most recent first, and the max_results parameter limits the number of Tweets returned per page
    search_results = client.search_recent(
        query=query, 
        max_results=max_results,
        tweet_fields="author_id,public_metrics",
    )
    
    return search_results 

if __name__ == "__main__":
    tags = ["#climateaction"]
    tag_activity_dict = {}
    for tag in tags:
        query = f"{tag} lang:en -is:retweet"
        activity_dict = tag_activity(query)
        tag_activity_dict[tag] = activity_dict
        
    most_active_tag = max(tag_activity_dict, key=lambda k: sum(tag_activity_dict[k]['counts']))
    print(f"The most active tag is: {most_active_tag}")
    
    search_results = pull_tweets(f"{most_active_tag} lang:en -is:retweet", max_results=10)
    
    with open('tweet_log/120825_tweets_climate.json', 'w', encoding='utf-8') as f:
        for page in search_results:
            # The Twitter API v2 returns the Tweet information and the user, media etc. separately
            # so we use expansions.flatten to get all the information in a single JSON
            result = expansions.flatten(page)
            for tweet in result:
                # Write each tweet as a JSON object on a new line
                f.write(json.dumps(tweet) + '\n')
            break  # Break after the first page to limit output