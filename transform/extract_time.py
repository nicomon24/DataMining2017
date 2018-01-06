# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

MONTHS = ['JUL','AUG','SEP','OCT','NOV','DEC']
PAY_MONTHS = ['PAY_' + m for m in MONTHS]
PAY_AMT_MONTHS = ['PAY_AMT_' + m for m in MONTHS]
BILL_AMT_MONTHS = ['BILL_AMT_' + m for m in MONTHS]
mon_dict = {'pay_delay':PAY_MONTHS, 'pay_amt':PAY_AMT_MONTHS, 'bill': BILL_AMT_MONTHS}

def build_features(df, prefix, p_weights = [1,1,1,1,1,1]):
    months = mon_dict[prefix]
    t = df[months[0]] * p_weights[0]
    for i in range(1,len(months)):
        t = t + df[months[i]] * p_weights[i]
    df[prefix + '_P'] = t / sum(p_weights)
    tmp = pd.DataFrame()
    for i in range(len(months)-1):
        month = months[i]
        tmp['DIFF_' + month] = df[months[i+1]] - df[month]
    df[prefix + '_D'] = tmp.mean(axis=1)
