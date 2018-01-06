# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

#Bills and payments normalization

#Normalization method: ( value - minval ) / ( maxval - minval )
def normalize(dataframe, prefix, normalizer='LIMIT_BAL'):
    mon_list = ['DEC','NOV','OCT','SEP','AUG','JUL']
    if prefix == 'DIFF':
        mon_list = ['DEC','NOV','OCT','SEP','AUG']
    fields = [prefix + '_' + month for month in mon_list]
    for x in fields:
        dataframe[x] = dataframe[x]/normalizer
