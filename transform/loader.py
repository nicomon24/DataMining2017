'''
    This module loads different dataset and apply basic transformation to them
        - Change DEFAULT PAYMENT JAN to TARGET
        - Change BIRTH_DATE to AGE

'''
import pandas as pd
import numpy as np

def load_dataset(filename, delimiter=',', format_ = '%d/%m/%Y'):
    df = pd.read_csv(filename, sep = delimiter, converters={'BIRTH_DATE': lambda x: "/".join(x.split('T')[0].split('-')[::-1]) if x != '#N/A' else '01/01/2017'})
    # Transform BIRTH_DATE to AGE
    df['AGE'] = (pd.to_datetime('today').year - pd.to_datetime(df['BIRTH_DATE'], format = format_).dt.year)
    df.drop(['BIRTH_DATE'],inplace=True, axis=1)
    # Rename TARGET
    df['TARGET'] = df['DEFAULT PAYMENT JAN']
    df.drop(['DEFAULT PAYMENT JAN'], inplace=True, axis=1)
    return df


