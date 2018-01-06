import pandas as pd
import numpy as np
import math


def removeZeros(df):
    mon_list = ['DEC','NOV','OCT','SEP','AUG','JUL']
    cols = ['BILL_AMT_' + month for month in mon_list] + ['PAY_AMT_' + month for month in mon_list]
    #Statistical analysis on zeros
    sliced = df[cols].copy()
    sliced['SquareSum'] = sum([sliced[x]**2 for x in sliced])
    mask = (sliced['SquareSum'] > 0.0001) | (sliced['SquareSum'] < -0.0001)
    ndf = df[mask]
    return ndf

# -*- coding: utf-8 -*-

