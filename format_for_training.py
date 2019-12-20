from __future__ import print_function

import math

import numpy as np
import pandas as pd
import re
import os
from pprint import pprint
def format_df(dataset):
    # print(dataset.columns)
    # exit()
    data_list = dataset["String_Generated"].to_list()
    label_list = dataset["Labels"].to_list()
    special_char = []
    length_data = []
    number_numeric = []
    ratio_numeric = []
    ratio_alpha = []
    ratio_special = []
    number_alpha = []
    slash_count = []
    rfq = []
    rfp = []
    rfa = []
    rft = []
    ifb = []
    itp = []
    itb = []
    space_count = []

    for item in data_list:
        item = str(item)
        length_item = (len(item))
        length_data.append(length_item)
        a = (len(re.findall('[^\w\*]',item))- len(re.findall(" ",item))-len(re.findall('/', item)))
        special_char.append(a)
        b = (len(re.findall(" ",item)))
        space_count.append(b)
        c = (len(re.findall('/', item)))
        slash_count.append(c)
        d = (len(re.findall('RFQ', item.upper())))
        rfq.append(d)
        e = (len(re.findall('RFP', item.upper())))
        rfp.append(e)
        f = (len(re.findall('RFA', item.upper())))
        rfa.append(f)
        g = (len(re.findall('IFB', item.upper())))
        ifb.append(g)
        h = (len(re.findall('ITB', item.upper())))
        itb.append(h)
        i = (len(re.findall('ITP', item.upper())))
        itp.append(i)
        k = (len(re.findall('RFT', item.upper())))
        rft.append(k)
        j = (a/length_item)
        ratio_special.append(j)
        
        d=l=0
        
        for c in item:
            if c.isdigit():
                d=d+1
            elif c.isalpha():
                l=l+1
            else:
                pass
        number_numeric.append(d)
        ratio_numeric.append(d/length_item)
        number_alpha.append(l) 
        ratio_alpha.append(l/length_item)


    my_dictionary = {'Length_Data' : length_data,
                'Number_Numeric' : number_numeric,
                'Number_Alpha' : number_alpha,
                'Special_Char': special_char,
                'Space_Count': space_count,
                'RFP_count': rfp,
                'RFQ_count': rfq,
                'RFA_count': rfa,
                'RFT_count': rft,
                'IFB_count': ifb,
                'ITP_count': itp,
                'ITB_count': itb,
                'Ratio_numeric':ratio_numeric,
                'Ratio_alpha':ratio_alpha,
                'Ratio_special':ratio_special,
                'Slash_count': slash_count,
                'label' : label_list
                }
    # print(my_dictionary['Ratio_special'])
    df = pd.DataFrame(my_dictionary)
    return df

def linear_scale(series):
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 2.0
    return series.apply(lambda x:((x - min_val) / scale) - 1.0)
