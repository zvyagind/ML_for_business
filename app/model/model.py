# sourcery skip: avoid-builtin-shadow
import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool, cv

dir = os.getcwd()

X_train = pd.read_csv(dir + "/app/model/dataset/fraudTrain.csv")
y_train = X_train.pop('target')

X_test =  pd.read_csv(dir+ "/app/model/dataset/fraudTest.csv")
y_test = X_test.pop('is_fraud')

cat = CatBoostClassifier()

params = {"iterations": 100,
          "depth": 2,
          "loss_function": "RMSE",
          "verbose": False}

cat_features = []


def transform(df):
        # drop columns with missing values
        df.dropna(axis=1, how='any', inplace=True)
     
        # transform object columns to categorical
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype('category')        
            
        #transform object columns to numbers
        df.select_dtypes(include=['category']).apply(lambda x: x.cat.codes)
        
        #fill missing values with zero
        columns = ['last_new_job']
        df['']
        
        # drop unnecessary columns
        columns = ['enrollee_id']
        df.drop(columns = columns, inplace = True)
        return df

cv_dataset = Pool(data=X_train,
                  label=y_train)

scores = cv(cv_dataset,
            params,
            fold_count=2, 
            plot="True")

grid = {'learning_rate': [0.03, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 9]}

grid_search_result = cat.grid_search(grid, X=X_train, y=y_train, plot=True)



def load_model(model_path):
    	# load the pre-trained model
	global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)
	print(model)

