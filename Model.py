# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 01:00:40 2023

@author: Sarra Ben Brahim
"""

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.corpus import stopwords
    import scipy.sparse
    import re
    from sklearn.linear_model import Ridge, Lasso
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import mean_squared_error
    import joblib
    import pickle
    import logging
    nltk.download('stopwords')
    nltk.download('wordnet')
    # Tokenize the data
    nltk.download('punkt')
    pass
except Exception as e:
    logging.error("Exception occurred", exc_info=True)
    
# Check if the 'punkt' tokenizer is downloaded. If not, download it.
if not nltk.download("punkt", quiet=True):
    # Handle the case where the download may fail due to network issues, etc.
    logging.error("Unable to download the 'punkt' tokenizer. Please check your internet connection and try again.")

# Create a tokenizer object for tokenizing the text.
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    
def process_text(text):
    """
    This function removes stop words of the germqn and english dictionnary from the text
    
    :param txt : text
    """
    german_stop_words = set(stopwords.words('german'))
    english_stop_words= set(stopwords.words('english'))
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    # Remove stop words from the text
    words = [word for word in words if (word.lower() not in german_stop_words) and (word.lower() not in english_stop_words)]
    text = ' '.join(words)
    return text

def clean_dataset(df): 
    """
    This function clean and tokenize the text data in the 'query' column
    It calculates a relevance score for each search query based on a heuristic.
    It adds this as a new column to the DataFrame.
    
    :param df : dataset stored in dataframe
    """
    logging.info('Cleaning the dataset ..')
    #preprocess the data 
    df['query'] = df['query'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower().strip()))
    # Tokenize the data
    df['query'] = df['query'].apply(lambda x: ' '.join(tokenizer.tokenize(x)))
    df['processed_query'] = df['query'].apply(process_text)  
    #relevance score calcultation 
    df["searches_per_month_norm"] = df["searches_per_month"] /max(df["searches_per_month"])
    df["success_rate_norm"] = df["success_rate"]
    df["conversion_rate_norm"] = df["conversion_rate"]
    # calculate the relevance score based on the heuristic
    df["relevance_score"] =  (df["searches_per_month_norm"]  * df["success_rate_norm"]  * df["conversion_rate_norm"])
    # Rescale the new column to have a range of [0, 1]
    df['relevance_score'] = (df['relevance_score'] - df['relevance_score'].min()) / (df['relevance_score'].max() - df['relevance_score'].min())
    # sort the dataset by relevance score in descending order
    df = df.sort_values("relevance_score", ascending=False)
    # Save the preprocessed dataframe to a new CSV file
    df.to_csv('preprocessed_data.csv', index=False)
    return df 

def split_data(df): 
    """
    This function splits the dataframe into training (80%) and testing (20%) .
    
    :param df : dataset stored in dataframe
    """
    logging.info('Spliting the dataset into train and test..')
    
    data = df
    train_data = data.sample(frac=0.8, random_state=42)
    test_data = data.drop(train_data.index)
    # Extract the queries and relevance scores from the training and testing sets
    train_queries = train_data['processed_query'].tolist()
    train_scores = train_data['relevance_score'].tolist()
    test_queries = test_data['query'].tolist()
    test_scores = test_data['relevance_score'].tolist()
    return train_queries,train_scores,test_queries, test_scores

def feature_engineering(train_queries, test_queries): 
    """
    This function  calculates the TF-IF for training and testing queries and stores them in sparse matrix.
    :param train_queries : queries of the training set
    :param train_queries : queries of the testing set
    """
    logging.info('Calculating TF-IDF for training and testing queries  ..')

    # Preprocess the training and testing queries using TF-IDF
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_queries)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    train_tfidf = vectorizer.transform(train_queries)
    test_tfidf = vectorizer.transform(test_queries)
    #transform into sparse matrix
    train_features = scipy.sparse.csr_matrix(train_tfidf)
    test_features = scipy.sparse.csr_matrix(test_tfidf)
    return train_features, test_features,vectorizer

def train_model(a,iterations, train_features,train_scores ):
    """
    This function  trains the model (Lasso) to predict the relevance.
    :param a : alpha of the Lasso model
    :param train_features : features for the training set
    :param train_scores : the target to predict (relevance score)
    
    """
    logging.info('Training the model : Lasso to predict the relevance score .. ')
    # Train a Ridge regression model on the training set 
    regressor = Lasso(alpha=a, max_iter=iterations)
    regressor.fit(train_features, train_scores)
    #Save the trained model to a file
    joblib.dump(regressor, 'Lasso.pkl')
    return regressor

def evaluate_model( model, test_features,test_scores ):
    """
    This function  evaluates the model performance for predicting the relevance score based on MSE and RMSE.
    :param model : queries of the training set
    :param test_features : features for the testing set
    :param test_scores : relevance score for the testing set
    
    """
    logging.info('Evaluating the model ..')
    y_pred = model.predict(test_features)
    test_mse = mean_squared_error(test_scores, y_pred)
    test_rmse = np.sqrt(test_mse)
    logging.info('Mean Square Error for test: %f', test_mse)
    logging.info('Root Mean Square Error for test: %f', test_rmse)
    return test_mse, test_rmse

def test_new_query(model,vectorizer, new_query):  
    """
    This function  test the model performance for predicting the relevance score on new query.
    :param model : model that will predict the relevance score
    :param vectorizer : this will transform the query into a vector
    :param new_query : new query to calculate its relevance score
    
    """    
    logging.info('Testing my model on new query ..')
    # Make predictions on new data
    query_vec = vectorizer.transform([new_query])
    test_predictions = regressor.predict(query_vec)
    logging.info('The predicted relevance score for %s: %f', new_query, predicted_relevance_score)
    return test_predictions[0]
   


if __name__ == "__main__":
    
    # Set up the logging configuration to write messages to a file
    logging.basicConfig(filename='Model.log',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    # Add a StreamHandler to the root logger to write messages to the console output
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
        
    logging.info('Reading the data')
   
    df = pd.read_csv('df_queries.csv')
    
    df = clean_dataset(df)
   
    train_queries,train_scores,test_queries, test_scores=split_data(df)
    
    train_features, test_features, vectorizer = feature_engineering(train_queries=train_queries, test_queries=test_queries)
    
    regressor= train_model(a=0.1,iterations=10000, train_features=train_features,train_scores= train_scores)

    test_mse, test_rmse = evaluate_model( model=regressor, test_features=test_features,test_scores=test_scores ) 

    predicted_relevance_score = test_new_query (model=regressor ,vectorizer=vectorizer, new_query="iphone 12" )
    
