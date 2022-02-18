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
            data[col] = (data[col]-data[col].mean())/data[col].std() # vladin komentar -> Z normalisation

    enc = OneHotEncoder(drop='if_binary')
    enc_data = pd.DataFrame(enc.fit_transform(data[categorical_cols]).toarray())

    data = data.join(enc_data)
    output = data['SalePrice']
    data = data.drop(columns = categorical_cols + ['SalePrice'])    

    return data, output


def realpath_folder():
    curfile = os.path.realpath(__file__)
    index = curfile.rfind("\\")
    curfolder = curfile[0:index]
    return curfolder


data = pd.read_csv(realpath_folder()+"\\"+'train.csv')
data, result = preprocess(data)

 
split = 0.8*data.shape[0]
x = data.iloc[:int(split),:].to_numpy()
val_x = data.iloc[int(split):,:].to_numpy()
y = result.iloc[:int(split)].to_numpy()
val_y = result.iloc[int(split):].to_numpy()

lmbd = 20
I = np.identity(x.shape[1])

b = np.matmul(np.matmul(np.linalg.pinv(np.matmul(x.T,x) + I*lmbd),x.T),y)

y_pred = np.matmul(val_x, b)

n = val_y.shape[0]
mae = sum(abs(val_y - y_pred))
mae = mae / n

mape = sum(abs((val_y - y_pred) / val_y))
mape = mape / n

print('mae =', mae)
print('mape =', mape)