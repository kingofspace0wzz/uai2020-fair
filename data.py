import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms

class CSVDataset(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(CSVDataset, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()

    def _normalize(self, tensors):
        pass

def load_adult(path, nogender):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'martial_status', 'occupation', 'relationship', 'race', 'sex',
                    'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']
    input_data = (pd.read_csv(path, names=column_names,
                              na_values="?", sep=r'\s*,\s*', engine='python')
                  )
    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    sensitive_attribs = ['sex']
    Z = (input_data.loc[:, sensitive_attribs]
         .assign(sex=lambda df: (df['sex'] == 'Male').astype(int)))

    # targets; 1 when someone makes over 50k , otherwise 0
    y = (input_data['target'] == '>50K').astype(int)
    
    # features; note that the 'target' columns are dropped
    X = (input_data
        .drop(columns=['target', 'sex'])
        .fillna('Unknown')
        .pipe(pd.get_dummies, drop_first=True)
        )
    # X.insert(0, "sex", Z['sex'])
        

    print("features X: {} samples, {} attributes".format(X.shape[0], X.shape[1]))
    print("targets y: {} samples".format(y.shape))
    print("sensitives Z: {} samples, {} attributes".format(Z.shape[0], Z.shape[1]))
    return X, y, Z

def get_adult(path, batch_size, nogender=False, test_size=1./3):
    X, Y, Z = load_adult(path, nogender)
    # X_test, y_test, Z_test = load_adult('adult.test', nogender)
    # split into train/test set
    (X_train, X_test, y_train, y_test,
    Z_train, Z_test) = train_test_split(X, Y, Z, test_size=test_size,
                                        stratify=Y, random_state=7)
    # standardize the data
    scaler = StandardScaler().fit(X_train)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), 
                                            columns=df.columns, index=df.index)
    X_train = X_train.pipe(scale_df, scaler) 
    X_test = X_test.pipe(scale_df, scaler)

    train_data = CSVDataset(X_train, y_train, Z_train)
    test_data = CSVDataset(X_test, y_test, Z_test)
  
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)
    return train_loader, test_loader

def load_german(path):
    column_names = ['status', 'duration', 'history', 'purpose', 'amount',
                    'savings', 'employment', 'income', 'sex', 'guarantors',
                    'residence', 'property', 'age', 'installment', 'housing',
                    'credits', 'job', 'liable', 'telephone', 'foreign', 'target']
    input_data = (pd.read_csv(path, names=column_names,
                              na_values="?", sep=r'\s*', engine='python')
                  )
    sensitive_attribs = ['age']
    # Z =(input_data.loc[:, sensitive_attribs].assign(age=lambda df: (int(df['age']) >= 45).astype(int)))
    Z = (input_data['age'].astype(int) >= 44).astype(int)
    Z = (Z.fillna('Unknown').pipe(pd.get_dummies, drop_first=True))
    y = (input_data['target'] == 1).astype(int)

    X = (input_data
            .drop(columns=['target', 'age'])
            .fillna('Unknown')
            .pipe(pd.get_dummies, drop_first=True))

    print("features X: {} samples, {} attributes".format(X.shape[0], X.shape[1]))
    print("targets y: {} samples".format(y.shape))
    print("sensitives Z: {} samples, {} attributes".format(Z.shape[0], Z.shape[1]))
    return X, y, Z

def get_german(path, batch_size, test_size=0.5):
    X, Y, Z = load_german(path)
    # split into train/test set
    (X_train, X_test, y_train, y_test,
    Z_train, Z_test) = train_test_split(X, Y, Z, test_size=test_size,
                                        stratify=Y, random_state=7)

    # standardize the data
    scaler = StandardScaler().fit(X_train)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), 
                                            columns=df.columns, index=df.index)
    X_train = X_train.pipe(scale_df, scaler) 
    X_test = X_test.pipe(scale_df, scaler)

    train_data = CSVDataset(X_train, y_train, Z_train)
    test_data = CSVDataset(X_test, y_test, Z_test)

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)
    return train_loader, test_loader

def load_crime(path, sensitive='blackPerCap'):
    column_names = ['{}'.format(i) for i in range(127)]
    column_names[-1] = 'target'
    column_names[7] = 'racePctBlack'
    column_names[27] = 'blackPerCap'
    column_names[66] = 'PctNotSpeakEnglWell'
    input_data = (pd.read_csv(path, names=column_names,
                              na_values="?", sep=r'\s*,\s*', engine='python')
                  )
    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    sensitive_attribs = [sensitive]
    Z = (input_data.loc[:, sensitive_attribs]
         .assign(blackPerCap=lambda df: (df[sensitive].astype(int) >= 0.2).astype(int)))
   
    # targets; 1 when someone makes over 50k , otherwise 0
    y = (input_data['target'].astype(int) >= 0.24).astype(int)
    
    # features; note that the 'target' columns are dropped
    X = (input_data
        .drop(columns=['target', sensitive])
        .fillna('Unknown')
        .pipe(pd.get_dummies, drop_first=True)
        )
        # X.insert(0, "sex", Z['sex'])
        

    print("features X: {} samples, {} attributes".format(X.shape[0], X.shape[1]))
    print("targets y: {} samples".format(y.shape))
    print("sensitives Z: {} samples, {} attributes".format(Z.shape[0], Z.shape[1]))
    return X, y, Z

def get_crime(path, batch_size, test_size=0.2):
    X, Y, Z = load_crime(path)
    # split into train/test set
    (X_train, X_test, y_train, y_test,
    Z_train, Z_test) = train_test_split(X, Y, Z, test_size=test_size,
                                        stratify=Y, random_state=7)

    # standardize the data
    scaler = StandardScaler().fit(X_train)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), 
                                            columns=df.columns, index=df.index)
    X_train = X_train.pipe(scale_df, scaler) 
    X_test = X_test.pipe(scale_df, scaler)

    train_data = CSVDataset(X_train, y_train, Z_train)
    test_data = CSVDataset(X_test, y_test, Z_test)

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)
    return train_loader, test_loader

def load_bank(path, nogender):
    column_names = ["age","job","marital","education","default","housing","loan","contact","month",
                    "day_of_week","duration","campaign","pdays","previous","poutcome","emp.var.rate",
                    "cons.price.idx","cons.conf.idx","euribor3m","nr.employed","y"]
    input_data = (pd.read_csv(path, names=column_names,
                              na_values="?", sep=r'\s*,\s*', engine='python')
                  )
    # sensitive attributes; we identify 'race' and 'sex' as sensitive attributes
    
    sensitive_attribs = ['age']
    
    Z = (input_data.loc[:, sensitive_attribs]
         .assign(age=lambda df: (df['age'].astype(int) >= 35).astype(int)))

    # targets; 1 when someone makes over 50k , otherwise 0
    y = (input_data['y'] == 'no').astype(int)
    # features; note that the 'target' columns are dropped
    X = (input_data
        .drop(columns=['y', 'age'])
        .fillna('Unknown')
        .pipe(pd.get_dummies, drop_first=True)
        )
        # X.insert(0, "sex", Z['sex'])
        

    print("features X: {} samples, {} attributes".format(X.shape[0], X.shape[1]))
    print("targets y: {} samples".format(y.shape))
    print("sensitives Z: {} samples, {} attributes".format(Z.shape[0], Z.shape[1]))
    return X, y, Z

def get_bank(path, batch_size, nogender=False, test_size=0.2):
    X, Y, Z = load_bank(path, nogender)
    # X_test, y_test, Z_test = load_adult('adult.test', nogender)
    # split into train/test set
    (X_train, X_test, y_train, y_test,
    Z_train, Z_test) = train_test_split(X, Y, Z, test_size=test_size,
                                        stratify=Y, random_state=7)
    # standardize the data
    scaler = StandardScaler().fit(X_train)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), 
                                            columns=df.columns, index=df.index)
    X_train = X_train.pipe(scale_df, scaler) 
    X_test = X_test.pipe(scale_df, scaler)

    train_data = CSVDataset(X_train, y_train, Z_train)
    test_data = CSVDataset(X_test, y_test, Z_test)
  
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_adult('adult.data', 5)
    data = train_loader.dataset
    print(data[:3][0].size())

    total = 0
    l = 0
    # for i, (x, y, z) in enumerate(train_loader):
    #     total += z.size(0)
    #     l += y.sum().item()
    #     x = torch.cat((z[:, 1].unsqueeze(1), x), dim=-1)
    # print(x[:5])
    # print(l / total)

    # sampler = torch.utils.data.RandomSampler(train_loader)
    
    # train_loader, test_loader = get_crime('communities.data', 5)
    # train_loader, test_loader = get_bank('bank-additional-full.csv', 5)
    # train_loader, test_loader = get_german('german.data', 100)
    # total = 0
    # l = 0
    # f = 0
    # for x, y, z in train_loader:
    #     total += z.size(0)
    #     f += z.sum().item() 
    #     l += y.sum().item()
    
    # print(f / total)
    # print(l / total)