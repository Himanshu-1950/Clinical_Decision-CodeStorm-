import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(path):
    df = pd.read_csv(path)
    df['gender'] = df['gender'].map({'M':1,'F':0})

    X = df.drop('disease', axis=1)
    y = df['disease']

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, le, scaler, X.shape[1]
