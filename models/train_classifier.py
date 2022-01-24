import sys
import pandas as pd
import numpy as np 

from sqlalchemy import create_engine
import re
import nltk
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath):
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql("SELECT * FROM DisasterText", engine)
    df['related'].replace(2, 1, inplace=True)
    X = df['message']
    y = df.iloc[:, 3:]
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    pipeline2 = Pipeline([
        
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    'clf__estimator__n_estimators': [50, 100],
    'clf__estimator__min_samples_split': [2, 3],
    }
    cv = GridSearchCV(pipeline2, param_grid=parameters)
    
    return cv
    
   
def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    for i in range(len(Y_test.columns)):
        print(Y_test.columns[i])
        print(classification_report(np.array(Y_test)[i], y_pred[i]))
        i += 1
    
    
    labels = np.unique(y_pred)
    Y_test = np.array(Y_test)
    confusion_mat = confusion_matrix(Y_test.argmax(axis=1), y_pred.argmax(axis=1), labels=labels)
    accuracy = (y_pred == Y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)
    
    return model.best_params_

def save_model(model, model_filepath):
    pickle.dump(model, open('classifier.pkl', 'wb'))


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
        
       
        #model = pickle.load(open("models/classifier.pkl", 'rb')) Model was trained for >1 hour and loading to prevent retraining
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()