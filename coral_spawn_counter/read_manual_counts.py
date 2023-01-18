#! /usr/bin/env python3

# read in csv table of manual counts

import pandas as pd
import os
from datetime import datetime





def read_manual_counts(file):
    df = pd.read_csv(os.path.join(filepath, filename))

    # datetime 
    # time of manual counts
    # convert to datetimes
    df['Datetime'] = df['Date'] + '-' + df['Time']
    dt = pd.to_datetime(df['Datetime'], format='%Y/%m/%d-%H:%M')

    # manual counts
    mc = df['Manual Count']

    # time water
    # time of switch/placement of CSLICS into water
    dwf = df['Date'][df['Time into water'].notna()] + '-' + df['Time into water'][df['Time into water'].notna()]
    tw = pd.to_datetime(dwf, format='%Y/%m/%d-%H:%M')

    return dt, mc, tw


if __name__ == "__main__":
    
    filepath = '/media/dorian/DT4TB/2022_NovSpawning/20221113_AMaggieTenuis/cslics04/metadata'
    filename = '20221113_ManualCounts_AMaggieTenuis_Tank4-Sheet1.csv'
    file = os.path.join(filepath, filename)
    
    dt, mc, tw = read_manual_counts(file)
    
    print(dt)
    print(mc)
    print(tw)
    import code
    code.interact(local=dict(globals(), **locals()))