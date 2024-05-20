import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# Function to preprocess text (remove punctuation, lowercase, etc.)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Function to collect user data from JSON
def collect_user_data_from_json(json_input):
    try:
        data = json.loads(json_input)
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return data
        else:
            return None
    except json.JSONDecodeError:
        return None

# Function to find similar descriptions
def find_similar_descriptions(user_description, sample_data):
    # Create a list to store descriptions
    description_list = [item["description"] for item in sample_data]

    # Preprocess all descriptions
    preprocessed_descriptions = [preprocess_text(desc) for desc in description_list]

    # Tokenize and vectorize the descriptions using TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_descriptions)

    # Preprocess the user's description
    user_description_processed = preprocess_text(user_description)

    # Vectorize the user's description
    user_tfidf = tfidf_vectorizer.transform([user_description_processed])

    # Calculate cosine similarity between user's description and all descriptions in the list
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

    # Get the indices of the three most similar descriptions
    sorted_indices = similarity_scores.argsort()[::-1][:7]

    # Retrieve the three most similar descriptions and their cosine similarity scores
    results = []
    for i in sorted_indices:
        result = {
            "description": description_list[i],
            "ID": sample_data[i]["ID"],
            "min_amount": sample_data[i]["min_amount"],
            "max_amount": sample_data[i]["max_amount"],
            "similarity_score": similarity_scores[i]
        }
        results.append(result)

    return results

@app.route('/', methods=['POST'])
def similarity_endpoint():
    json_input = request.form.get('json_input')
    user_description = request.form.get('user_description')

    if not json_input or not user_description:
        return jsonify({"error": "Missing json_input or user_description"}), 400

    sample_data = collect_user_data_from_json(json_input)
    if sample_data is None:
        return jsonify({"error": "Invalid JSON format"}), 400

    similar_descriptions = find_similar_descriptions(user_description, sample_data)

    return jsonify(similar_descriptions)

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0', port=5000)
