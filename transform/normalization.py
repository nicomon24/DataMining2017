'''
    This module loads different dataset and apply basic transformation to them
        - Change DEFAULT PAYMENT JAN to TARGET
        - Change BIRTH_DATE to AGE

'''
import pandas as pd
import numpy as np

MONTHS = ['JUL','AUG','SEP','OCT','NOV','DEC']
PAY_MONTHS = ['PAY_' + m for m in MONTHS]
PAY_AMT_MONTHS = ['PAY_AMT_' + m for m in MONTHS]
BILL_AMT_MONTHS = ['BILL_AMT_' + m for m in MONTHS]

# Remove negative values from PAY, set to 0
def remove_pay_negatives(data):
    for month in PAY_MONTHS:
        data[month] = data[month].clip(lower=-1)
    return data

def set_pay_negatives(data):
    for month in PAY_MONTHS:
        data[month] = data[month].clip(lower=-1)
    return data

def normalize(data, strategy='limit_bal'):
    if (strategy == 'limit_bal'):
        # Strategy: limit_bal. Normalize row-wise with respect to own limit_bal
        for month in PAY_AMT_MONTHS:
            data[month] = data[month] / data['LIMIT_BAL']
        for month in BILL_AMT_MONTHS:
            data[month] = data[month] / data['LIMIT_BAL']  
        return data
    elif (strategy == 'max'):
        # Strategy max. Normalize referring to the max value everywhere
        return data
    return data

def normalize_col_wise(df):
    for month in PAY_AMT_MONTHS:
        data[month] = data[month] / data[month].max()
    for month in BILL_AMT_MONTHS:
        data[month] = data[month] / data[month].max()  
        return data