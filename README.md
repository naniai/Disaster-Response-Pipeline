# Disaster Response Pipeline Project

### 描述:
该项目是Udacity与Figure Eight合作开发的数据科学纳米学位的项目。初始数据集包含预先标记的推文和灾难应对机构收到来自受害人群的求助信息。该项目的目的是构建一种对消息进行分类的自然语言处理工具。

该项目分为以下几节：

1. 数据处理，ETL管道可从源中提取数据，清除数据并将其保存在适当的数据库结构中

2. 机器学习管道可训练能够对文本进行分类的模型

3. Web App实时显示模型结果。

### 执行程序：
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### 附加材料：
在data and models文件夹中，您可以找到两个jupyter notebook，它们将帮助您逐步了解模型的工作方式：

1. ETL准备笔记本：了解有关已实现的ETL管道的所有信息

2. ML管道准备笔记本：看一下使用NLTK和Scikit-Learn开发的机器学习管道

您可以使用ML Pipeline Preparation Notebook重新训练模型或通过专用的Grid Search部分对其进行调整。
