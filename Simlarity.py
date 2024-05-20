import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data from the CSV file
csv_file = 'services.csv'
data = pd.read_csv(csv_file)

# Tokenize and vectorize the descriptions using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['description'])


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


if __name__ == "__main__":
    user_description = input("Enter a description: ")

    similar_descriptions, similar_jobs, similar_scores = find_similar_descriptions(user_description)

    print("Four most similar descriptions with their respective jobs and cosine similarity scores:")
    for i, (desc, job, score) in enumerate(zip(similar_descriptions, similar_jobs, similar_scores), 1):
        print(f"{i}. {desc} (Job: {job}) (Cosine Similarity: {score:.4f})")
