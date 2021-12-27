#%%

import pandas as pd
import numpy as np
import yfinance as yf
import bs4 as bs
import pickle
import requests
import datetime as dt
from datetime import datetime
import os
from tqdm.notebook import tqdm
import csv
from dateutil import parser
from decimal import Decimal
import math
import re
#import pandas_datareader.data as pdr


#%%

def read_file2(filepath):

    meta_rows= datetime.strptime(re.compile(r'\d+').findall(filepath)[0],'%Y%m%d').strftime("%Y-%m-%d")
    return meta_rows

def read_file(filepath):
    META_DATA_ROWS = 2  # Header data starts at line 4
    meta_rows = []
    calls_and_puts = []

    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        for row, cells in enumerate(reader):
            if row < META_DATA_ROWS:
                meta_rows.append(cells)                      
    return meta_rows


def get_dt_current(meta_rows):
    """
    Extracts time information.

    :param meta_rows: 2D array
    :return: parsed datetime object
    """
    # First cell of second row contains time info
    #date_time_row = '2021-03-22'+' @ 09:46 ET' #VALIDACAO
    #date_time_row = meta_rows[0][0]+' @ 20:00 ET' #versao single
    date_time_row = meta_rows+' @ 20:00 ET' #versao FINAL

    # Format text as ET time string
    current_time = date_time_row.strip()\
        .replace('@ ', '')\
        .replace('ET', '-05:00')\
        .replace(',', '')

    dt_current =  parser.parse(current_time)
    return dt_current

def parse_expiry_and_strike2(text):
    """
    Extracts information about the contract data.

    :param text: the string to parse.
    :return: a tuple of expiry date and strike price
    """
    # Should expire at 4PM Chicago time?
    year, month, day = text.split('-')
    #expiry = '%s %s %s 3:00PM -05:00' % (year, month, day) #VALIDACAO
    expiry = '%s %s %s 4:00PM -05:00' % (year, month, day)
    dt_object = parser.parse(expiry)    

    """
    Third friday SPX standard options expire at start of trading
    8.30 A.M. Chicago time. FAZ SENTIDO????????????
    """
    if is_third_friday(dt_object):
        dt_object = dt_object.replace(hour=8, minute=30)

    return dt_object

def is_third_friday(dt_object):
    return dt_object.weekday() == 4 and 15 <= dt_object.day <= 21

def get_near_next_terms(df,first_bracket,dt_current):
    dt_near = None
    dt_next = None
    for g_idx, group in df.groupby(['expirationDate'])[['strike', 'bid_x', 'ask_x', 'bid_y', 'ask_y']]:
        g_idx=parse_expiry_and_strike2(g_idx)
        delta = g_idx - dt_current
        if delta.days > first_bracket:
            # Skip non-fridays
            if g_idx.weekday() != 4:
                continue
            # Save the near term date
            if dt_near is None:
                dt_near = g_idx            
                continue
            # Save the next term date
            if dt_next is None:
                dt_next = g_idx            
                break
    #evitar o groupby ou for e fazer apenas loc a datas unique.
    return dt_near, dt_next

# Calculate the required minutes
def calc_minutes(dt_current,dt_near,dt_next,expiry_days):

    dt_start_year = dt_current.replace(
        month=1, day=1, hour=0, minute=0, second=0)
    dt_end_year = dt_start_year.replace(year=dt_current.year+1)

    N_t1 = (dt_near-dt_current).total_seconds() // 60
    N_t2 = (dt_next-dt_current).total_seconds() // 60
    N_30 = expiry_days * 24 * 60
    N_365 = (dt_end_year-dt_start_year).total_seconds() // 60

    t1 = N_t1 / N_365
    t2 = N_t2 / N_365

    return t1, t2, N_365, N_30, N_t1, N_t2

# Calc forward Underlying level
def determine_forward_level(df, r, t):
    """
    Calculate the forward underlying level.

    :param df: pandas DataFrame for a single option chain
    :param r: risk-free interest rate for t
    :param t: time to settlement in years
    :return: Decimal object
    """
    min_diff = min(df['diff'])
    pd_k = df[df['diff'] == min_diff]
    k = pd_k.index.values[0]

    call_price = pd_k.loc[k, 'x_mid']
    put_price = pd_k.loc[k, 'y_mid']

    f1=k + math.exp(r*t)*(call_price-put_price)

    return f1

# Required forward strike prices

def find_k0(df, f):
    return df[df.index<f].tail(1).index.values[0]

# Determining strike price boundaries
def find_lower_and_upper_bounds2(df,k0):

    new_low = df.copy()
    new_upp = df.copy()

    #LOWER LIMIT

    new_low.loc[:,'zero_bid']=(new_low['bid_y'] != 0).astype(int).cumsum()
    new_low.loc[:,'zero_bid_cum']=new_low.groupby(['zero_bid']).cumcount(ascending=False)+1
    #aux=new.sort_values(by=['zero_bid_cum']).reset_index()

    if 2 in new_low.loc[(new_low.index<k0)&(new_low['bid_y'] == 0),'zero_bid_cum'].to_list():
        first_index_aux=new_low.index.get_loc(new_low[(new_low['zero_bid_cum'] == 2) & (new_low['bid_y']==0)& (new_low.index<k0)].index[-1])
        k_lower=new_low.index[first_index_aux+2]

    else:
        k_lower=new_low.index[0]

    #k_lower=new_low.loc[new_low.index > first_index].index[0]
    #new_low=new_low.loc[new_low.index > first_index].index[0]
    #k_low=new_low.loc[new_low.index < k0_near].index[0]


    #UPPER LIMIT

    new_upp.loc[:,'zero_bid']=(new_upp['bid_x'] != 0).astype(int)[::-1].cumsum()
    new_upp.loc[:,'zero_bid_cum']=new_upp.groupby(['zero_bid']).cumcount()+1

    if 2 in new_upp.loc[(new_upp.index>k0)&(new_upp['bid_x'] == 0),'zero_bid_cum'].to_list():
        first_index_aux=new_upp.index.get_loc(new_upp[(new_upp['zero_bid_cum'] == 2) & (new_upp['bid_x']==0)& (new_upp.index>k0)].index[0])
        k_upper=new_upp.index[first_index_aux-2]
    else:
        k_upper=new_upp.index[-1]

    #k_upper=new_upp.loc[new_upp.index <first_index].index[-1]

    return (k_lower, k_upper)

def tabulate_contrib_by_strike_faster(df, k0, k_lower, k_upper, r, t):
    #Slice df_near
    new = df.loc[k_lower:k_upper].copy()

    # Conditions to separate out-of-the-money puts and calls
    condition1 = new.index < k0
    condition2 = new.index > k0
    
    new.loc[condition1,'Option Type']='Put'
    new.loc[condition2,'Option Type']='Call'
    new.loc[ k0,'Option Type']='atm'

    new=new[((new.bid_x!= 0) & (new.bid_y!= 0)) | (new.index == k0)]
    #new=new[new.bid_x!= 0 & (new.index > k0)]

    new.loc[new.index < k0,'mid']=new['y_mid']
    new.loc[new.index > k0,'mid']=new['x_mid']
    new.loc[new.index == k0,'mid']=(new['y_mid']+new['x_mid'])/2

    new.loc[new.iloc[1:-1].index,'dk']=np.array((new.iloc[2:].index - new.iloc[:-2].index) / 2)
    new.loc[new.index[0],'dk']=new.index[1] - new.index[0]
    new.loc[new.index[-1],'dk']=new.index[-1] - new.index[-2]

    new['contrib']=new['dk']/new.index**2*math.exp(r*t)*new['mid']

    new=new[['Option Type','mid','contrib']]

    return new

def calculate_volatility(pd_contrib, t, f, k0):
    """
    Calculate the volatility for a single-term option

    :param pd_contrib: pandas DataFrame containing 
        contributions by strike
    :param t: time to settlement of the option
    :param f: forward index level
    :param k0: immediate strike price below the forward level
    :return: volatility as Decimal object
    """
    term_1 = 2/t*pd_contrib['contrib'].sum()
    term_2 = 1/t*(f/k0 - 1)**2
    return term_1 - term_2

def calculate_vix_index(t1, volatility_1, t2, 
                        volatility_2, N_t1, N_t2, N_30, N_365):
    inner_term_1 = t1*volatility_1*(N_t2-N_30)/(N_t2-N_t1)
    inner_term_2 = t2*volatility_2*(N_30-N_t1)/(N_t2-N_t1)
    sqrt_terms = math.sqrt((inner_term_1+inner_term_2)*N_365/N_30)
    return 100 * sqrt_terms

#%%

def iv_calc(mat,path):


    """     near=pd.read_csv('validacaonear.csv',delim_whitespace=True )
        near['expirationDate']='2021-04-16'
        near['x_mid']=(near['bid_x']+near['ask_x']) / 2
        near['y_mid']=(near['bid_y']+near['ask_y']) / 2
        near['diff']=abs(near['y_mid']-near['x_mid'])
        near=near[['expirationDate','strike','bid_x','ask_x','x_mid','bid_y','ask_y','y_mid','diff']]

        next=pd.read_csv('validacaonext.csv',delim_whitespace=True )
        next['expirationDate']='2021-04-23'
        next['x_mid']=(next['bid_x']+next['ask_x']) / 2
        next['y_mid']=(next['bid_y']+next['ask_y']) / 2
        next['diff']=abs(next['y_mid']-next['x_mid'])
        next=next[['expirationDate','strike','bid_x','ask_x','x_mid','bid_y','ask_y','y_mid','diff']]

        df2=pd.concat([near,next]).reset_index(drop=True) """

    try:
        df = pd.read_csv(path,index_col=0 )

    

        call_df=df[df['CALL'] == True]
        put_df=df[df['CALL'] == False]

        myorder = [8, 1, 0, 2, 3, 4, 5, 6, 7, 9, 10]
        col_order=[call_df.columns[i] for i in myorder]


        call_df=call_df.loc[:,col_order]
        call_df['expirationDate']=pd.to_datetime(call_df.iloc[:,0], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
        #call_df['First_col']=call_df.loc[:,call_df.columns[0:3]].apply(lambda x: ' '.join(x.astype(str)),axis=1)
        #call_df['First_col']=call_df.loc[:,call_df.columns[0:3]].values.tolist()

        put_df=put_df.loc[:,col_order]
        #put_df['expirationDate']=pd.to_datetime(put_df.iloc[:,0], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
        #put_df['First_col']=put_df.loc[:,put_df.columns[0:3]].apply(lambda x: ' '.join(x.astype(str)),axis=1)

        df2=pd.merge(call_df, put_df, on=['expirationDate','strike'], how='inner')

        #Remove NANs
        df2=df2.dropna()

        df2['x_mid']=(df2['bid_x']+df2['ask_x']) / 2
        df2['y_mid']=(df2['bid_y']+df2['ask_y']) / 2
        df2['diff']=abs(df2['y_mid']-df2['x_mid'])
        df2=df2[['expirationDate','strike','bid_x','ask_x','x_mid','bid_y','ask_y','y_mid','diff']]
        #df2=df2.loc[:, ['strike','First_col_x','bid_x','ask_x','First_col_y','bid_y','ask_y']]

        #call_df.groupby(['expirationDate']).nth(0)
        #pd.pivot_table(call_df, index='expirationDate', values=['bid','ask'])
        #call_df.groupby(level=['expirationDate']).groups
        #call_df.groupby(['expirationDate'])[['strike', 'bid', 'ask']]
        #df2.groupby(['expirationDate'])[['expirationDate','strike','bid_x','ask_x','x_mid','bid_y','ask_y','y_mid','diff']].get_group('2021-04-09').set_index('strike')
        #call_df.pivot( level=['expirationDate'])
        #call_df.pivot_table(index='expirationDate',values=['bid'])
        #call_df.groupby('expirationDate').apply(lambda x: x.sort_values('strike'))


        """ 
        id_group=df.groupby(['Category','Level'])

        for g_idx, group in id_group:
            for r_idx, row in group.iterrows():
                if (((row['Metric_LHS'] > group['Metric_RHS']).any())
                    & (row['Metric_LHS'] > row['Baseline'])):
                    df.loc[r_idx, 'Opportunity?'] = 1
        """



        meta_rows=read_file2(path)
        dt_current =  get_dt_current(meta_rows)
        print(dt_current)
        dt_near, dt_next=get_near_next_terms(df2,mat-7,dt_current)

        t1, t2, N_365, N_30, N_t1, N_t2=calc_minutes(dt_current,dt_near,dt_next,mat)

        print('N_30 (min):', N_30)
        print('N_t1 (min):', N_t1)
        print('N_t2 (min):', N_t2)
        print('t1:%.5f'%t1)
        print('t2:%.5f'%t2)
        
        r = 0.0286/100

        #Near
        df_near = df2.groupby(['expirationDate'])[['strike', 'bid_x', 'ask_x','x_mid', 'bid_y', 'ask_y','y_mid','diff']].get_group(dt_near.strftime('%Y-%m-%d')).set_index('strike')
        f1 = determine_forward_level(df_near, r, t1)
        print('f1:', f1)
        k0_near = find_k0(df_near, f1)
        print('k0_near:', k0_near)
        (k_lower_near, k_upper_near) =find_lower_and_upper_bounds2(df_near, k0_near)
        print(k_lower_near, k_upper_near)
        pd_contrib_near = tabulate_contrib_by_strike_faster(df_near, k0_near, k_lower_near, k_upper_near, r, t1)
        volatility_near = calculate_volatility(pd_contrib_near, t1, f1, k0_near)
        print('volatility_near:', volatility_near)

        #Next
        df_next = df2.groupby(['expirationDate'])[['strike', 'bid_x', 'ask_x','x_mid', 'bid_y', 'ask_y','y_mid','diff']].get_group(dt_next.strftime('%Y-%m-%d')).set_index('strike')
        f2 = determine_forward_level(df_next, r, t2)
        print('f2:', f2)
        k0_next = find_k0(df_next, f2)
        print('k0_next:', k0_next)
        (k_lower_next, k_upper_next) =find_lower_and_upper_bounds2(df_next,k0_next)
        pd_contrib_next = tabulate_contrib_by_strike_faster(df_next, k0_next, k_lower_next, k_upper_next, r, t2)
        volatility_next = calculate_volatility(pd_contrib_next, t2, f2, k0_next)
        print('volatility_next:', volatility_next)

        vix = calculate_vix_index(t1, volatility_near, t2,volatility_next, N_t1, N_t2,N_30, N_365)

        print('At', dt_current, 'the VIX is', vix)
    
    except Exception as e:
        print(str(e))

    #return vix;

#iv_calc(30, pathfile) #Define number of IV days (30 menas 30-day IV)

#%%
folders=['options_dfs_'+re.compile(r'\d+').findall(f)[0]+'-'+re.compile(r'\d+').findall(f)[1] for f in os.listdir(os.path.dirname(os.path.abspath(__file__))) if f.startswith('options_dfs_')]
folder_list=list(map(lambda suffix: os.path.dirname(__file__)+'\\'+suffix, folders))
suffix='AAPL.csv'
paths=list(map(lambda x: x+'\\'+suffix, folder_list))
list(map(lambda x: iv_calc(30, x), paths))




#%% Tabulating contributions by strike prices -OBSOLETO

def calculate_contrib_by_strike(delta_k, k, r, t, q):
    return (delta_k / k**2)*math.exp(r*t)*q

def find_prev_k(k, i, k_lower, df, bid_column):
    """
    Finds the strike price immediately below k 
    with non-zero bid.

    :param k: current strike price at i
    :param i: current index of df
    :param k_lower: lower strike price boundary of df
    :param bid_column: The column name that reads the bid price.
        Can be 'put_bid' or 'call_bid'.
    :return: strike price as Decimal object.
    """    
    if k <= k_lower:
        k_prev = df.index[i]
        return k_prev

    # Iterate backwards to find put bids           
    k_prev = 0
    prev_bid = 0
    steps = 1
    while prev_bid == 0:                                
        k_prev = df.index[i-steps]
        prev_bid = df.loc[k_prev][bid_column]
        steps += 1

    return k_prev

def find_next_k(k, i, k_upper, df, bid_column):
    """
    Finds the strike price immediately above k 
    with non-zero bid.

    :param k: current strike price at i
    :param i: current index of df
    :param k_upper: upper strike price boundary of df
    :param bid_column: The column name that reads the bid price.
        Can be 'put_bid' or 'call_bid'.
    :return: strike price as Decimal object.
    """    
    if k >= k_upper and df.index[-1]!=k_upper:
        k_next = df.index[i]
        return k_next
    elif k >= k_upper and df.index[-1]==k_upper:
        k_next = df.index[i]
        return k_next

    k_next = 0
    next_bid = 0
    steps = 1
    while next_bid == 0:
        k_next = df.index[i+steps]
        next_bid = df.loc[k_next][bid_column]
        steps += 1

    return k_next

def tabulate_contrib_by_strike(df, k0, k_lower, k_upper, r, t):
    """
    Computes the contribution to the VIX index
    for every strike price in df.

    :param df: pandas DataFrame containing the option dataset
    :param k0: forward strike price index level
    :param k_lower: lower boundary strike price
    :param k_upper: upper boundary strike price
    :param r: the risk-free interest rate
    :param t: the time to expiry, in years
    :return: new pandas DataFrame with contributions by strike price
    """
    COLUMNS = ['Option Type', 'mid', 'contrib']
    pd_contrib = pd.DataFrame(columns=COLUMNS)

    for i, k in enumerate(df.index):
        mid, bid, bid_column = 0, 0, ''
        if k_lower <= k < k0:
            option_type = 'Put'
            bid_column = 'bid_y'
            mid = df.loc[k]['y_mid']
            bid = df.loc[k][bid_column]
        elif k == k0:
            option_type = 'atm'
            bid_column = 'bid_y' #adicionado por mim
            mid = (df.loc[k]['y_mid']+df.loc[k]['x_mid'])/2 #adicionado por mim
            bid = df.loc[k][bid_column] #adicionado por mim
        elif k0 < k <= k_upper:
            option_type = 'Call'
            bid_column = 'bid_x'
            mid = df.loc[k]['x_mid']
            bid = df.loc[k][bid_column]
        else:
            continue  # skip out-of-range strike prices

        if bid == 0:
            continue  # skip zero bids

        k_prev = find_prev_k(k, i, k_lower, df, bid_column)
        k_next = find_next_k(k, i, k_upper, df, bid_column)
        if k==k_lower or k==k_upper:
            delta_k=abs(k_next-k_prev)

        else:
            delta_k = (k_next-k_prev)/2
            
        #print(i,k,k_prev,k_next, delta_k)
        contrib = calculate_contrib_by_strike(delta_k, k, r, t, mid)
        pd_contrib.loc[k, COLUMNS] = [option_type, mid, contrib]

    return pd_contrib



#%% OBSOLETO

def find_lower_and_upper_bounds(df, k0):
    """
    Find the lower and upper boundry strike prices.

    :param df: the pandas DataFrame of option chain
    :param k0: the forward strike price
    :return: a tuple of two Decimal objects
    """
    # Find lower bound
    otm_puts = df[df.index<k0].filter(['bid_y', 'ask_y'])
    k_lower = 0
    for i, k in enumerate(otm_puts.index[::-1][:-2]):
        k_lower = k
        put_bid_t1 = otm_puts.iloc[-i-1-1]['bid_y'] # penultimo
        put_bid_t2 = otm_puts.iloc[-i-1-2]['bid_y'] # ultimio
        if put_bid_t1 == 0 and put_bid_t2 == 0:
            break
        if put_bid_t2 == 0:
            k_lower = otm_puts.index[-i-1-1]

    # Find upper bound
    otm_calls = df[df.index>k0].filter(['bid_x', 'ask_x'])    
    k_upper = 0
    for i, k in enumerate(otm_calls.index[:-2]):
        call_bid_t1 = otm_calls.iloc[i+1]['bid_x'] # penultimo
        call_bid_t2 = otm_calls.iloc[i+2]['bid_x'] # ultimo
        if call_bid_t1 == 0 and call_bid_t2 == 0:
            k_upper = k
            break
        elif call_bid_t2 == 0:
            k_upper = otm_calls.index[i+1]
        else:
            k_upper = otm_calls.index[i+2]

    return (k_lower, k_upper)



#%%

#write metadata
today = dt.datetime.today().strftime('%Y-%m-%d')  
with open('AAPL_{}.csv'.format(today), "w") as f:
    f.write(today+'\n')
#append df to existing file filled by metadata
df.to_csv('AAPL_{}.csv'.format(today), index=False, mode='a')
#################

# %%


