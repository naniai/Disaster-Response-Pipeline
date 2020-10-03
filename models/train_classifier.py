import sys
import nltk
import pickle
nltk.download(['punkt','wordnet','averaged_perceptron_tagger'])
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    加载数据
    Input:
        database_filepath-> SQLite数据库的路径
    Output:
        X->输入特征数据
        Y->标签DataFrame
        category_names->用于数据可视化（应用程序）
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages_pie', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    """
    清洗和分词
    Input:
        text ->信息列表（英文）
    Output:
        clean_tokens->标记化文本，用于ML建模清理
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    创建机器学习管道
    Input:
        message 列
    Output:
        输出分类结果，分类结果属于该数据集中的 36 个类
    """
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    """
    parameters = {'vect__max_df': (0.75, 1.0),
            'vect__max_features': (None, 5000),
            'tfidf__use_idf': (True, False)
             }

    model = GridSearchCV(pipeline, param_grid=parameters,cv=5, verbose=4, n_jobs=-1)
    """
    
    return model
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
    评估模型功能
    此功能将ML管道应用于测试集并输出
    模型性能（准确性和f1score）
    
    Input：
        model-> Scikit ML管道
        X_test->测试集
        Y_test->测试集
        category_names->标签名称（多输出）
    """
    y_pred = model.predict(X_test)
    y_pred_pd = pd.DataFrame(y_pred, columns = Y_test.columns)
    
    for column in Y_test.columns:
        print('\n')
        print('FEATURE: {}\n'.format(column))
        print(classification_report(Y_test[column],y_pred_pd[column]))
    
    accuracy = (y_pred == Y_test).mean().mean()
    print("Accuracy:", accuracy)
    
    pass    

def save_model(model, model_filepath):
    """
    保存模型
    将训练好的模型另存为Pickle文件，以供以后加载。
    Input：
        model-> GridSearchCV或Scikit Pipelin对象
        model_filepath->保存为 .pkl文件 的目标路径
    Output:
        None
    """
    with open(model_filepath, 'wb') as pickle_file:
        pickle.dump(model, pickle_file)

def main():
    """
    此功能适用于机器学习管道：
        1）从SQLite数据库中提取数据
        2）在训练集上训练ML模型
        3）评估测试集上的模型性能
        4）将经过训练的模型另存为Pickle
    """
    
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