# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 20:36:54 2023

@author: 
"""

try:
    import logging
    from flask import Flask, request, jsonify
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import joblib
    import pickle
    pass
except Exception as e:
    logging.error("Exception occurred", exc_info=True)

def load_data():
    """Load the dataset, preprocessed data, model and vectorizer."""
    df = pd.read_csv("df_queries.csv")
    df2 = pd.read_csv("preprocessed_data.csv")
    relevance_scores = df2["relevance_score"]
    relevance_model = joblib.load('Lasso.pkl')
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    tfidf_matrix = vectorizer.transform(df["query"])
    return df, relevance_scores, relevance_model, vectorizer, tfidf_matrix


app = Flask(__name__)

df, relevance_scores, relevance_model, vectorizer, tfidf_matrix = load_data()


@app.route("/")
def index():
    """Return a message to confirm that the service is running."""
    return "Query similarity service is running!"


@app.route("/api/similar_queries", methods=["GET"])
def get_similar_queries():
    """Get the top 10 most similar queries to the input query."""
    query = request.args.get("query")
    if not query:
        return jsonify({
            "User Message ": "No Result",
            "similar_queries": []
        })
    query_vec = vectorizer.transform([query])
    relevance_score = relevance_model.predict(query_vec)[0]
    relevant_query_indices = [
        i for i, score in enumerate(relevance_scores) if score >= relevance_score
    ]
    relevant_queries = df["query"].iloc[relevant_query_indices]
    relevant_tfidf_matrix = tfidf_matrix[relevant_query_indices]
    sim_scores = cosine_similarity(query_vec, relevant_tfidf_matrix)
    similar_query_indices = sim_scores.argsort()[0][-11:-1][::-1]
    similar_queries = list(df["query"].iloc[similar_query_indices])
    logging.info("Top 10 similar queries to '%s': %s", query, similar_queries)
    return jsonify({
        "User Message": "Top 10 matching results",
        "similar_queries": similar_queries
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

