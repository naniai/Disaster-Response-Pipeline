import sys
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def load_data(messages_filepath, categories_filepath):
    """
    加载数据
    
    Input:
        包含消息的csv文件->messages_filepath
        包含类别的csv文件->categories_filepath
    Output:
        将数据加载为 Pandas DataFrame
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, how='inner')
    return df


def clean_data(df):
    """
    清洗数据
    
    Input:原始数据 Pandas DataFrame
    Output:清洗后的数据 Pandas DataFrame
    """
    #分割categories
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    #转换类别值至数值 0 或 1
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str.get(-1)
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    #替换 df categories 类别列
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    #删除重复行
    df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    df.duplicated().sum()
    return df


def save_data(df, database_filename):
    """
    保存整理后的数据集为 sqlite databse
    
    Input:清洗后的数据 Pandas DataFrame
    Output:数据库文件
    """
    engine = create_engine('sqlite:///'+database_filename)
    Session = sessionmaker(bind=engine)
    session = Session()  #invokes sessionmaker.__call__()
    session.execute('DROP TABLE IF EXISTS messages_pie')

    df.to_sql('messages_pie', engine, index=False)
    

def main():
    """
    处理数据
    
    此函数执行ETL管道：
         1）从.csv提取数据
         2）数据清理和预处理
         3）将数据加载到SQLite数据库
    """
   
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