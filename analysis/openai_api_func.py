from openai import OpenAI
import os
from dotenv import load_dotenv
import random
import time

load_dotenv()
api_key = os.getenv("OPEN_AI_API_KEY")

def beyond_binary_sentiment_analysis(text, max_retries=3):
    '''
    Perform beyond binary sentiment analysis using OpenAI's API GPT-5-nano model.
    Parameters:
    text (str): The text to analyze.
    max_retries (int): The maximum number of retries for the API call in case of failure. Default is 3.
    Returns:
    str: The emotion classified for the tweet, one of 'Rage', 'Triumph', 'Despair', 'Empathy', 'Hope', or 'Confusion'.
    If the API call fails, it will retry up to max_retries times with exponential backoff.
    '''
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[
                    {
                        "role": "system",
                        "content": "Classify the tweet emotion as exactly one of: 'Rage', 'Triumph', 'Despair', 'Empathy', 'Hope', or 'Confusion'. Respond with only the single word."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text
                            },
                        ]
                    }
                ]
            )
            
            print("Response content:", response.choices[0].message.content)
            print("Usage - Total tokens:", response.usage.total_tokens)
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying API call in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                raise Exception(f"OpenAI API call failed after {max_retries} attempts: {e}")

def secondary_sentiment_analysis(text, primary_sentiment, max_retries=3):
    '''
    Perform secondary sentiment analysis using OpenAI's API GPT-5-nano model.
    Parameters:
    text (str): The text to analyze.
    primary_sentiment (str): The primary sentiment of the text.
    max_retries (int): The maximum number of retries for the API call in case of failure. Default is 3.
    Returns:
    str: The sentiment classified for the tweet, one of 'Positive', 'Negative', or 'Neutral'.
    If the API call fails, it will retry up to max_retries times with exponential backoff.
    '''
    sent_dict = {
        'Rage': ['Indignation', 'Contempt', 'Fury'],
        'Despair': ['Helplessness', 'Grief', 'Resignation'],
        'Confusion': ['Bewilderment', 'Frustration', 'Disbelief'],
        'Empathy': ['Compassion', 'Shared Joy', 'Sympathy'],
        'Hope': ['Anticipation', 'Resilience', 'Encouragement'],
        'Triumph': ['Elation', 'Defiance', 'Empowerment']
    }
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[
                    {
                        "role": "system",
                        f"content": "Classify the tweet sentiment as exactly one of: {sent_dict[primary_sentiment][0]}, {sent_dict[primary_sentiment][1]}, or {sent_dict[primary_sentiment][2]}. Respond with only the single word."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": text
                            },
                        ]
                    }
                ]
            )
            
            print("Response content:", response.choices[0].message.content)
            print("Usage - Total tokens:", response.usage.total_tokens)
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying API call in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                raise Exception(f"OpenAI API call failed after {max_retries} attempts: {e}")

def main():
    tweet_text = "The world is falling behind with meeting climate targets, and the current lack of effort is evidenced by increased suffering and poverty."
    emotion = beyond_binary_sentiment_analysis(tweet_text)
    secondary_emotion = secondary_sentiment_analysis(tweet_text, emotion)
    print(f"The emotion classified for the tweet is: {emotion}")
    print(f"The secondary emotion classified for the tweet is: {secondary_emotion}")

if __name__ == "__main__":
    main()