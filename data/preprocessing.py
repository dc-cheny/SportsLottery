import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def ThreeDProcessor(df):
    # deal with result column and row_date column
    news = df["result"].str.split(" ", n=2, expand=True)
    df['result_0'], df['result_1'], df['result_2'] = news[0], news[1], news[2]
    df['row_date'] = df['row_date'].apply(lambda x: ''.join(x.split('-')))

    # deal with two zuxuan columns
    scaler = MinMaxScaler()
    df['zuxuan3_num'] = df['zuxuan3_num'].apply(lambda x: int(x) if x != '\xa0' else np.nan)
    df['zuxuan3_num'] = scaler.fit_transform(df['zuxuan3_num'].astype('float').values.reshape(-1, 1))
    df['zuxuan6_num'] = df['zuxuan6_num'].apply(lambda x: int(x) if x != '\xa0' else np.nan)
    df['zuxuan6_num'] = scaler.fit_transform(df['zuxuan6_num'].astype('float').values.reshape(-1, 1))
    df['is_zuxuan3'] = df.apply(lambda x: 0 if pd.isnull(x['zuxuan3_num']) else 1, axis=1)
    df['zuxuan_num'] = df.apply(lambda x: x['zuxuan6_num'] if pd.isnull(x['zuxuan3_num']) else x['zuxuan3_num'], axis=1)

    df.drop('result', axis=1, inplace=True)
    df.drop('zuxuan3_num', axis=1, inplace=True)
    df.drop('zuxuan6_num', axis=1, inplace=True)

    # transform str to float and standardize
    df['num'] = df['num'].astype(float)
    df['sum'] = df['sum'].astype(float)
    df['zhixuan_num'] = df['zhixuan_num'].astype(float)
    df['row_date'] = df['row_date'].astype(float)
    df['result_0'] = df['result_0'].astype(float)
    df['result_1'] = df['result_1'].astype(float)
    df['result_2'] = df['result_2'].astype(float)

    df['num'] = scaler.fit_transform(df['num'].values.reshape(-1, 1))
    df['sum'] = scaler.fit_transform(df['sum'].values.reshape(-1, 1))
    df['zhixuan_num'] = scaler.fit_transform(df['zhixuan_num'].values.reshape(-1, 1))
    df['row_date'] = scaler.fit_transform(df['row_date'].values.reshape(-1, 1))
    df['result_0'] = scaler.fit_transform(df['result_0'].values.reshape(-1, 1))
    df['result_1'] = scaler.fit_transform(df['result_1'].values.reshape(-1, 1))
    df['result_2'] = scaler.fit_transform(df['result_2'].values.reshape(-1, 1))

    return df
