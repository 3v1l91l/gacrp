import numpy as np
import pandas as pd
import os
import json

json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']

def process_json(df):
    for column in json_cols:
        print(column)
        c_load = df[column].apply(json.loads)
        c_list = list(c_load)
        c_dat = json.dumps(c_list)

        json_df = pd.read_json(c_dat)
        json_df.columns = [column + '_' + x for x in json_df.columns]
        for c in json_df.columns:
            if isinstance(json_df.loc[0, c], dict):
                json_df.drop(c, axis=1, inplace=True)

        json_df_nunique = json_df.nunique()
        if((json_df_nunique == 1).any()):
            json_df.drop(json_df_nunique[json_df_nunique == 1].index.values, axis=1, inplace=True)

        df = df.join(json_df)
        df = df.drop(column, axis=1)

    return df

def main():
    print('process train')
    train = pd.read_csv(os.path.join('..', 'input', 'train.csv'),
                        dtype={'date': str, 'fullVisitorId': str, 'sessionId':str})
    train = process_json(train)
    print('process test')
    test = pd.read_csv(os.path.join('..', 'input', 'test.csv'),
                        dtype={'date': str, 'fullVisitorId': str, 'sessionId':str})
    test = process_json(test)

    train.to_pickle(os.path.join('..', 'input', 'train_ext_json.pkl'))
    test.to_pickle(os.path.join('..', 'input', 'test_ext_json.pkl'))


if __name__ == '__main__':
    main()