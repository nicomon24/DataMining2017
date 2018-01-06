'''

    This module contains different encoding for the dataset
        - LabelEncoder
        - OneHotEncoder

'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

def label_encode(data):
    df = data
    # Encode SEX, check for NaN
    df['SEX'] = df.SEX.fillna('NaN')
    le_sex = LabelEncoder()
    df['SEX'] = le_sex.fit_transform(df.SEX)
    # Encode EDUCATION, check for NaN
    df['EDUCATION'] = df.EDUCATION.fillna('NaN')
    le_edu = LabelEncoder()
    df['EDUCATION'] = le_edu.fit_transform(df.EDUCATION)
    # Encode MARRIAGE, check for NaN
    df['MARRIAGE'] = df.MARRIAGE.fillna('NaN')
    le_marriage = LabelEncoder()
    df['MARRIAGE'] = le_marriage.fit_transform(df.MARRIAGE)
    return df

def onehot_encode(data):
    return pd.get_dummies(data, columns=['SEX','EDUCATION','MARRIAGE'])
