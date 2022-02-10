import sys
import pandas as pd
import numpy as np 

from sqlalchemy import create_engine # Interact with SQL database
import re # Import regular expression library
import nltk # Import Natural Language Toolkit
from nltk.tokenize import word_tokenize
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet'])
from sklearn.metrics import confusion_matrix

from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report # Reporting F1 score 
from sklearn.model_selection import GridSearchCV

import pickle # Library for saving and loading model file

def load_data(database_filepath):
    """ Load a previously created database. """
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql("SELECT * FROM DisasterText", engine)
    df['related'].replace(2, 1, inplace=True) # Data cleaning step to replace irrelevant 2's 
    X = df['message'] # Creating the feature dataframe
    y = df.iloc[:, 3:] # Creating the target dataframe
    category_names = y.columns # Saving the labels
    
    return X, y, category_names


def tokenize(text):
    """ Converting words to tokenized words. """
    text.lower() # Convert entire text to lowercase. 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = []
    for i in tokens:
        clean_tok = lemmatizer.lemmatize(i).lower().strip()
        clean_tokens.append(clean_tok)
    
    return cleaned_tokens


def build_model():
    """ Creating the model pipeline to allow for multiple parameter testing, and selecting best performing model. """
    pipeline = Pipeline([

        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [5],
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3)
    
    return cv
    
   
def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluation of model performance. """
    y_pred = model.predict(X_test)
    for i in range(len(Y_test.columns)):
        print(Y_test.columns[i])
        print(classification_report(np.array(Y_test)[i], y_pred[i])) # Reporting F1 scores
        i += 1
    
    
    labels = np.unique(y_pred)
    Y_test = np.array(Y_test)
    confusion_mat = confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1), labels=labels) # Constructing confusion matrix
    accuracy = (y_pred == Y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)
    
    return model.best_params_ # Returning best performing model

def save_model(model, model_filepath):
    """Saving the model in a Pickle file. """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) # Splitting training and testing data
        
        print('Building model...')
        model = build_model() 
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
       
        #model = pickle.load(open("models/classifier.pkl", 'rb')) Model was trained for >1 hour and loading to prevent retraining
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