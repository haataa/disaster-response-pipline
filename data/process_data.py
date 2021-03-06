import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load csv files and return a merged dataframe
    
    Args:
    messages_filepath: string. Filepath of message csv.
    categories_filepath: string. Filepath of categories of messages.
       
    Returns:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on ='id')
    return df

def clean_data(df):
    """Clean the dataframe 
    Args :
        df : dataframe to be cleaned
    Returns :
        df : cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[0:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    # convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    df.drop(columns=['related','child_alone'],inplace=True)
    return df
    



def save_data(df, database_filename):
    """Save dataframe into an SQLite database
    Args :
    df : dataframe. Data to be saved
    database_filename : string. Filename for output database.
    
    Returns :
    None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    # need to add chunksize or otherwise there will be too many sql mistake
    df.to_sql('messages', engine, index=False, if_exists="replace",chunksize=999)


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