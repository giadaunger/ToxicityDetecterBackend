from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import praw
import tweepy
import os
import re
import emoji
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from openai import OpenAI

load_dotenv()
app = Flask(__name__)
CORS(app) 

# Load the model and tokenizer
tokenizer = BertTokenizer.from_pretrained('model/toxicity_tokenizer') 
model = BertForSequenceClassification.from_pretrained('model/toxicity_model')  
model.eval()  

# Clean text
def clean_text(text):
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = emoji.replace_emoji(text, "")
    return text


# Open API

def gpt_explain_toxicity(text, flagged_labels):
    explanation_prompt = f"The following comment was flagged as potentially harmful:\n\n{text}\n\n"
    explanation_prompt += "Based on the following flags:\n"
    
    # Add each flagged label with the probability score
    for label, score in flagged_labels.items():
        explanation_prompt += f"- {label}: {score:.2f}\n"
    
    explanation_prompt += "Explain why this text might be harmful or toxic."

    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "You are a toxicity analysis assistant."},
                {"role": "user", "content": explanation_prompt}
            ],
            max_tokens=150,
            temperature=0.5
        )
        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        return f"Error generating explanation: {str(e)}"


# Reddit 
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT"),
)

def fetch_reddit_thread(url):
    try:
        submission = reddit.submission(url=url)
        submission.comments.replace_more(limit=0)
        title = clean_text(submission.title)
        author = clean_text(submission.author.name) if submission.author else "Unknown"
        comments = []
        
        for comment in submission.comments.list():
            cleaned_comment = clean_text(comment.body)
            toxicity = predict_toxicity(cleaned_comment)

            # Only append comments that are flagged with an explanation
            if toxicity["explanation"]:
                comments.append({
                    "text": cleaned_comment,
                    "toxicity": toxicity["prediction"],
                    "explanation": toxicity["explanation"]
                })
        
        # Print the output structure for debugging
        response_data = {
            "title": title,
            "author": author,
            "comments": comments
        }
        print("Reddit thread response:", response_data)  # Debugging line
        return response_data
    except Exception as e:
        print(f"Error fetching Reddit thread: {e}")
        return {"error": str(e)}



# Twitter
client = tweepy.Client(bearer_token=os.getenv("BEARER_TOKEN"))

def fetch_twitter_thread(tweet_id):
    try:
        tweet = client.get_tweet(tweet_id, tweet_fields=["conversation_id", "author_id"])
        conversation_id = tweet.data["conversation_id"]
        
        thread_tweets = client.search_recent_tweets(query=f"conversation_id:{conversation_id}", tweet_fields=["author_id", "text"])
        if not thread_tweets.data:
            return {"error": "No thread found for the given tweet"}

        thread = [tweet.text for tweet in thread_tweets.data]

        return {
            "main_tweet": tweet.data["text"],
            "thread": thread
        }
    except Exception as e:
        return {"error": str(e)}


def extract_tweet_id(tweet_url):
    tweet_id = tweet_url.split('/')[-1]
    print(f"Extracted tweet_id: {tweet_id}") 
    return tweet_id


# Prediciton
def predict_toxicity(text):
    # Clean and preprocess the text
    cleaned_text = clean_text(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding='max_length', max_length=128)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.sigmoid(logits).squeeze().tolist()  # Convert to probabilities
    
    # Define the labels
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    prediction = {label: prob for label, prob in zip(labels, probabilities)}
    
    threshold = 0.25
    flagged_labels = {label: score for label, score in prediction.items() if score >= threshold}

    # Generate explanation only if there are flagged labels
    explanation = None
    if flagged_labels:
        explanation = gpt_explain_toxicity(text, flagged_labels)

    return {"prediction": prediction, "explanation": explanation}


# Endpoint to classify comments
@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.json
    text = data.get('text', None)
    
    if text:
        prediction = predict_toxicity(text)
        return jsonify(prediction), 200
    else:
        return jsonify({"error": "No text provided"}), 400
    

# Endpoint to handle link submission
@app.route('/submit', methods=['POST'])
def receive_input():
    data = request.json 
    link = data.get('link')

    if not data:
        return jsonify({'error': 'No data received'}), 400
    
    if "reddit.com" in link:
        reddit_data = fetch_reddit_thread(link)
        return jsonify(reddit_data), 200
    elif "x.com" in link:
        tweet_id = extract_tweet_id(link)
        twitter_thread = fetch_twitter_thread(tweet_id)
        return jsonify(twitter_thread), 200
    else:
        return jsonify({'error': 'Invalid link or unsupported platform'}), 400

if __name__ == '__main__':
    app.run(debug=True)