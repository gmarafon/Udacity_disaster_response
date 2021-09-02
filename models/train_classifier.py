import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import time
from joblib import parallel_backend #to select the backend for the training
import pickle

from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

from xgboost import XGBClassifier
import xgboost as xgb

def load_data(database_filepath):
    '''
    Load data from categories SQLite file and return the data splitted into X, y and a list with the categories names
    Attributes:
    database_filepath: path for the SQLite file
    '''
    engine = create_engine('sqlite:///' + database_filepath)

    with engine.connect() as connection:
        df = pd.read_sql('SELECT * FROM categories', connection)

    #separate into fetures and labels
    X = df['message']
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = y.columns.to_list()

    return X, y, category_names


def tokenize(text):
    '''
    Clean, lemmatize and tokenize the text
    Attributes:
    text: Text to be tokenized
    '''
    #clean
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    
    #tokenize
    tokens = word_tokenize(text)
    
    #stop words
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    
    #initiate lammatizer
    lemmatizer = WordNetLemmatizer()
    
    #loop through each token and record
    clean_tokens = []
    for token in tokens:
        lemmatize = lemmatizer.lemmatize(token.lower().strip())
        clean_tokens.append(lemmatize)
        
    return clean_tokens


def build_model():
    '''
    Create a pipeline with the machine learning model and return it
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfid', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(XGBClassifier(use_label_encoder=False)))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the created model and return a df with the precision, recall and f1-score for each category
    and print it
    Attributes:
    model: model to be evaluated
    X_test: input test records
    Y_test: label test records
    category_names: list of categories to be evaluated
    '''

    y_pred = model.predict(X_test)

    #create the classification_report for each category passed and print it
    df_report = pd.DataFrame()
    for i, v in enumerate(Y_test):
        report = classification_report(Y_test.iloc[:,i], y_pred[:,i], output_dict=True, zero_division=0)
        report = pd.DataFrame(report).iloc[:3,3]
        df_report = pd.concat([df_report, report], axis=1)
        df_report.rename(columns={df_report.columns[i] : category_names[i]}, inplace=True)
    df_report = df_report.transpose()
    print(df_report)
    print('precision average: {}'.format(df_report.iloc[:,0].mean()))
    print('recall average: {}'.format(df_report.iloc[:,1].mean()))
    print('f1-score average: {}'.format(df_report.iloc[:,2].mean()))
    return None


def save_model(model, model_filepath):
    '''
    Save the model in a pickle file
    Attribues:
    model: model name
    model_filepath: path to save the pickle file
    '''
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)

    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        #commented out due to problem with custom models when saving pickle file
        #the pickle dumping was moved to pickle_dump.py. more info in the README
        #print('Saving model...\n    MODEL: {}'.format(model_filepath))
        #save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()