import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os


def categorical_col(data, col):
    if (len(data[col].unique()) < 20 or data[col].dtype.char == 'O'):
        return True
    return False
 
def numerical_col(data, col):
    if (len(data[col].unique()) >= 20 and data[col].dtype in [np.int64, np.float64]):
        return True
    return False
 
def preprocess(data):
    cols_to_drop = ['Id']
    
    data = data.drop(columns=cols_to_drop)
 
    year_cols = []
    for col in data.columns:
        if 'Yr' in col or 'Year' in col:
            year_cols.append(col)
    
    for col in year_cols:
        data[col] = data['YrSold'] - data[col]
        data[col] = data[col].fillna(data[col].median())
 
    data = data.drop(columns = ['YrSold'])
 
 
    categorical_cols = []
    data = data.reset_index()
    for col in data.columns:
        if categorical_col(data, col) and not col in year_cols:
            categorical_cols.append(col)
            data[col] = data[col].fillna('Missing')
 
    quantitative_cols = []  
    data = data.reset_index()
    for col in data.columns:
        if numerical_col(data, col) and col != 'SalePrice' and col not in year_cols:
            quantitative_cols.append(col)
            data[col] = data[col].fillna(0)
            data[col] = (data[col]-data[col].mean())/data[col].std()
 
  
    enc = OneHotEncoder(drop='if_binary')
    enc_data = pd.DataFrame(enc.fit_transform(data[categorical_cols].astype(str)).toarray())
 
    data = data.join(enc_data)
    data = data.drop(columns=categorical_cols)

    return data

def realpath_folder():
    curfile = os.path.realpath(__file__)
    index = curfile.rfind("\\")
    curfolder = curfile[0:index]
    return curfolder

df_train = pd.read_csv(realpath_folder()+"\\"+'train.csv')
df_test = pd.read_csv(realpath_folder()+"\\"+'test.csv')
y = df_train.pop('SalePrice')
 
df_concat = pd.concat([df_train, df_test])
 
df_processed = preprocess(df_concat)
df_train = df_processed.iloc[0:df_train.shape[0]]
df_test =  df_processed.iloc[df_train.shape[0]:]
 
 
x = df_train.to_numpy()
y = y.to_numpy()
val_x = df_test.to_numpy()
 
lmbd = 20
I = np.identity(x.shape[1])
 
b = np.matmul(np.matmul(np.linalg.pinv(np.matmul(x.T,x) + I*lmbd),x.T),y)
preds = np.matmul(val_x,b)


preds_df = pd.DataFrame(preds,columns=['SalePrice'])
preds_df.index = preds_df.index+1461
preds_df.to_csv(realpath_folder()+"\\"+'submission.csv', index=True)