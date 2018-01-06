'''
    This module deals with missing values. Possible strategies:
        - Remove rows with 1 missing value
        - TODO: interpolation and stuff
'''
import pandas as pd
import numpy as np
import math

def drop_missing(data):
    df = data.copy(deep=True)
    df.dropna(inplace=True)
    return df

def fill_missing(data, strategy='dummy'):
    if strategy == 'dummy':
        data.AGE.fillna(-1, inplace=True)
        data.SEX.fillna('NaN', inplace=True)
        data.EDUCATION.fillna('NaN', inplace=True)
        data.MARRIAGE.fillna('NaN', inplace=True)
    return data
