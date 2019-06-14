import sys
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from io import StringIO
import pandas as pd
import numpy as np
import re
import pickle

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report,accuracy_score


def load_data(database_filepath):
    """Load and return the needed data
    
    Args:
    database_filename: string. SQLite database filepath.
       
    Returns:
    X: dataframe. Features dataset.
    Y: dataframe. Labels dataset.
    category_names: list of strings. List containing category names.
    """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("messages", con=engine)
    #df.head()
    X = df['message'] 
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    category_names = df.columns[-34:]
    print(category_names)
    print(Y.head())
    return X,Y,category_names

def tokenize(text):
    """Normalize, tokenize and lemmatize text string
    
    Args:
    text: string. String to be cleaned
       
    Returns:
    words: list of strings. List containing clean word tokens
    """
    # step 1 clean text, remove punctuation and turn into lower cases
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    # tokenize 
    words = word_tokenize(text)
    # remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    # lemmatize word tokens
    words = [WordNetLemmatizer().lemmatize(w).lower().strip() for w in words]
    
    return words


def build_model():
    """Build a machine learning pipeline.
    Return:
        cv : GridSearch model
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'vect__min_df': [1, 5],
              'tfidf__use_idf':[True, False],
              'clf__estimator__n_estimators':[10, 20]}

    cv = GridSearchCV(pipeline, param_grid = parameters,verbose = 3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Shows model's performance on test data
    Args :
        model: trained model
        X_test: Test features
        Y_test: Test targets
        category_names: Target labels
    Return:
        None
    """

    # predict
    y_pred = model.predict(X_test)

    # print classification report
    print(classification_report(Y_test.values, y_pred, target_names=category_names))

    # print accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
    """Save the trained model in a pickle file
    Args :
        model : trained model.
        model_filepath : string. Filepath of the model to be saved
    Returns:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()