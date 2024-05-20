import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template_string

# Load the data from the CSV file
csv_file = 'services.csv'
data = pd.read_csv(csv_file)

# Tokenize and vectorize the descriptions using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['description'])

app = Flask(__name__)


def find_similar_descriptions(user_description):
    # Vectorize the user's description
    user_tfidf = tfidf_vectorizer.transform([user_description])

    # Calculate cosine similarity between user's description and all descriptions in the dataset
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

    # Get the indices of the four most similar descriptions
    sorted_indices = similarity_scores.argsort()[::-1][:4]

    # Retrieve the four most similar descriptions and their cosine similarity scores
    most_similar_descriptions = data.iloc[sorted_indices]['description'].tolist()
    most_similar_jobs = data.iloc[sorted_indices]['job'].tolist()
    most_similar_scores = similarity_scores[sorted_indices]

    return most_similar_descriptions, most_similar_jobs, most_similar_scores


@app.route('/')
def home():
    return render_template_string("""
    <html>
    <body>
        <h1>Welcome to the Similar Descriptions Service</h1>
        <p>Use the <code>/find_similar</code> endpoint to find similar descriptions.</p>
    </body>
    </html>
    """)


@app.route('/find_similar', methods=['POST'])
def find_similar():
    user_description = request.json.get('description')
    if not user_description:
        return jsonify({"error": "Description is required"}), 400

    similar_descriptions, similar_jobs, similar_scores = find_similar_descriptions(user_description)

    results = []
    for desc, job, score in zip(similar_descriptions, similar_jobs, similar_scores):
        results.append({
            "description": desc,
            "job": job,
            "similarity_score": score
        })

    return jsonify(results)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug=True)
