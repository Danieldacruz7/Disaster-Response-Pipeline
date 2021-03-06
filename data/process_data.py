import sys
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    """Loading CSV file data."""
    messages = pd.read_csv(messages_filepath) # Convert messages.csv to dataframe
    categories = pd.read_csv(categories_filepath) # Convert categories.csv to dataframe
    df = messages.set_index('id').join(categories.set_index('id'))
    
    categories = df['categories'].str.split(';',expand=True) # Delimiting columns by semi-colon 
    row = categories.iloc[1]
    category_colnames = []
    for i in row:
        i = i[:-2]
        category_colnames.append(i)
        
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].astype(str).apply(lambda x: int(x[-1]))
    
    df.drop(['categories'], axis=1, inplace=True)
    
    df = pd.concat([df, categories], axis=1, sort=True) # Combining to dataframes into one
    
    return df


def clean_data(df):
    """Cleaning data for database insertion. """
    df.drop_duplicates(keep=False, inplace=True) # Remove duplicates from data
    df['related'].replace(2, 1, inplace=True) # Replace 2's with 1's
    
    return df # Returned cleaned dataframe


def save_data(df, database_filename):
    """ Insert cleaned data into a database. """
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///data/DisasterResponse.db') # Creating SQL database
    df.to_sql('DisasterText', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()