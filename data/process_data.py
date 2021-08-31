import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load messages and categories files from a csv to a dataframe, merge them togheter and return the dataframe
    Attributes:
    messages_filepath = fullpath including filename
    categories_filepath = fullpath including filename
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='left', on='id')

    return df


def clean_data(df):
    '''
    Clean and transform the dataframe to expand the categories into individual columns
    Attributes:
    df = merged dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    names = categories.iloc[0,:].str[:-2].to_list()
    categories.columns = names
    categories = categories.apply(lambda x: x.str[-1], axis=1)
    categories = categories.apply(pd.to_numeric, axis=1)
    categories['related'].replace(2, 1, inplace=True) #replace 2 by 1
    
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    cleaned_df = pd.concat([df, categories], axis=1)

    #drop duplicates
    df.drop_duplicates('id', inplace=True, ignore_index=False)

    return cleaned_df


def save_data(df, database_filename):
    '''
    Save the DataFrame into a SQLite file
    Attributes:
    df = DataFrame to be stored
    database_filename = filename of the database to be stored
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df.to_sql('categories', engine, index=False, if_exists='replace')

    return None  


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
