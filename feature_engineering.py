
import numpy as np
import pandas as pd
from itertools import combinations
import os
import pytz
from datetime import datetime

def time_zone_converter(x):
    try:
        return pytz.country_timezones(x)[0]
    except AttributeError:
        return np.nan

def time_localizer(s):
    # format of series [time,zone]
    try:
        tz = pytz.timezone(s[1])
        return pytz.utc.localize(datetime.utcfromtimestamp(s[0]), is_dst=None).astimezone(tz)

    except:
        return np.nan

def map_timezone(x):
    try:
        global timezone_dict
        return timezone_dict[x]
    except KeyError:
        return 'UTC'

def remove_missing_vals(x):
    remove_list = ['(not set)', 'not available in demo dataset', 'unknown.unknown']
    if x in remove_list:
        return ''
    else:
        return x

def browser_mapping(x):
    browsers = ['chrome', 'safari', 'firefox', 'internet explorer', 'edge', 'opera', 'coc coc', 'maxthon', 'iron']
    if x in browsers:
        return x.lower()
    elif ('android' in x) or ('samsung' in x) or ('mini' in x) or ('iphone' in x) or ('in-app' in x) or (
        'playstation' in x):
        return 'mobile browser'
    elif ('mozilla' in x) or ('chrome' in x) or ('blackberry' in x) or ('nokia' in x) or ('browser' in x) or (
        'amazon' in x):
        return 'mobile browser'
    elif ('lunascape' in x) or ('netscape' in x) or ('blackberry' in x) or ('konqueror' in x) or ('puffin' in x) or (
        'amazon' in x):
        return 'mobile browser'
    elif '(not set)' in x:
        return x
    else:
        return 'others'

def adcontents_mapping(x):
    if ('google' in x):
        return 'google'
    elif ('placement' in x) | ('placememnt' in x):
        return 'placement'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'ad' in x:
        return 'ad'
    else:
        return 'others'

def device_mapping(x):
    if ('windows' in x):
        return 'windows'
    elif ('macintosh' in x):
        return 'macintosh'
    elif ('android' in x) or ('samsung' in x) or ('blackberry' in x):
        return 'android'
    elif ('ios' in x):
        return 'ios'
    elif ('linux' in x) or ('freebsd' in x) or ('openbsd' in x) or ('sunos' in x):
        return 'linux'
    else:
        return 'others'

def source_mapping(x):
    if ('google' in x):
        return 'google'
    elif ('youtube' in x):
        return 'youtube'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'yahoo' in x:
        return 'yahoo'
    elif 'facebook' in x:
        return 'facebook'
    elif 'reddit' in x:
        return 'reddit'
    elif 'bing' in x:
        return 'bing'
    elif 'quora' in x:
        return 'quora'
    elif 'outlook' in x:
        return 'outlook'
    elif 'linkedin' in x:
        return 'linkedin'
    elif 'pinterest' in x:
        return 'pinterest'
    elif 'ask' in x:
        return 'ask'
    elif 'siliconvalley' in x:
        return 'siliconvalley'
    elif 'lunametrics' in x:
        return 'lunametrics'
    elif 'amazon' in x:
        return 'amazon'
    elif 'mysearch' in x:
        return 'mysearch'
    elif 'qiita' in x:
        return 'qiita'
    elif 'messenger' in x:
        return 'messenger'
    elif 'twitter' in x:
        return 'twitter'
    elif 't.co' in x:
        return 't.co'
    elif 'vk.com' in x:
        return 'vk.com'
    elif 'search' in x:
        return 'search'
    elif 'edu' in x:
        return 'edu'
    elif 'mail' in x:
        return 'mail'
    elif 'ad' in x:
        return 'ad'
    elif 'golang' in x:
        return 'golang'
    elif 'direct' in x:
        return 'direct'
    elif 'dealspotr' in x:
        return 'dealspotr'
    elif 'sashihara' in x:
        return 'sashihara'
    elif 'phandroid' in x:
        return 'phandroid'
    elif 'baidu' in x:
        return 'baidu'
    elif 'mdn' in x:
        return 'mdn'
    elif 'duckduckgo' in x:
        return 'duckduckgo'
    elif 'seroundtable' in x:
        return 'seroundtable'
    elif 'metrics' in x:
        return 'metrics'
    elif 'sogou' in x:
        return 'sogou'
    elif 'businessinsider' in x:
        return 'businessinsider'
    elif 'github' in x:
        return 'github'
    elif 'gophergala' in x:
        return 'gophergala'
    elif 'yandex' in x:
        return 'yandex'
    elif 'msn' in x:
        return 'msn'
    elif 'dfa' in x:
        return 'dfa'
    elif '(not set)' in x:
        return '(not set)'
    elif 'feedly' in x:
        return 'feedly'
    elif 'arstechnica' in x:
        return 'arstechnica'
    elif 'squishable' in x:
        return 'squishable'
    elif 'flipboard' in x:
        return 'flipboard'
    elif 't-online.de' in x:
        return 't-online.de'
    elif 'sm.cn' in x:
        return 'sm.cn'
    elif 'wow' in x:
        return 'wow'
    elif 'baidu' in x:
        return 'baidu'
    elif 'partners' in x:
        return 'partners'
    else:
        return 'others'

def process_device(data_df):
    print("process device ...")
    device_cols = ['device_browser', 'device_deviceCategory', 'device_operatingSystem']
    for i in device_cols:
        for j in device_cols:
            if(i==j):
                continue
            data_df['fe_' + i + "_" + j] = data_df[i] + "_" + data_df[j]

    return data_df

def geo(data):
    print('geo')
    for i in ['geoNetwork_city', 'geoNetwork_continent', 'geoNetwork_country', 'geoNetwork_metro',
              'geoNetwork_networkDomain', 'geoNetwork_region', 'geoNetwork_subContinent']:
        for j in ['device_browser', 'device_deviceCategory', 'device_operatingSystem',
                  'ts_day_of_week', 'channelGrouping', 'trafficSource_medium']:
            data['fe_' + i + "_" + j] = data[i] + "_" + data[j]

    return data

def traffic_source(data):
    print('trafficSource')
    for i in ['trafficSource_source']:
        for j in ['device_browser', 'device_deviceCategory', 'device_operatingSystem',
                  'ts_day_of_week', 'channelGrouping', 'trafficSource_medium',
                  'geoNetwork_city', 'geoNetwork_continent', 'geoNetwork_country', 'geoNetwork_metro',
                  'geoNetwork_networkDomain', 'geoNetwork_region', 'geoNetwork_subContinent'
                  ]:
            data['fe_' + i + "_" + j] = data[i] + "_" + data[j]

    return data


def extract_time(df):
    # df['ts_day_of_week'] = df['date'].map(lambda x: x.weekday()).astype('str')
    df['ts_day_of_week'] = df['visit_ts'].map(lambda x: x.strftime('%A')).astype('str')
    df['ts_hour_of_day_int'] = df['visit_ts'].map(lambda x: x.strftime('%H')).astype('int8')
    # df['ts_day_of_month'] = df['visit_ts'].map(lambda x: x.strftime('%d')).astype('str')
    # df['ts_month'] = df['visit_ts'].map(lambda x: x.strftime('%d')).astype('str')

    return df

def timezone_conv(train, test):
    geocode_df = pd.read_pickle(os.path.join('..', 'input', 'geocodes_timezones.pkl'))
    train['_search_term'] = train['geoNetwork_city'].map(remove_missing_vals) + ' ' + train['geoNetwork_region'].map(
        remove_missing_vals) + ' ' + train['geoNetwork_country'].map(remove_missing_vals)
    test['_search_term'] = test['geoNetwork_city'].map(remove_missing_vals) + ' ' + test['geoNetwork_region'].map(
        remove_missing_vals) + ' ' + test['geoNetwork_country'].map(remove_missing_vals)

    global timezone_dict
    timezone_dict = dict(zip(geocode_df['search_term'], geocode_df['timeZoneId']))

    train['_timeZoneId'] = train['_search_term'].map(map_timezone)
    test['_timeZoneId'] = test['_search_term'].map(map_timezone)

    train['visit_ts'] = train[['visitStartTime', '_timeZoneId']].apply(time_localizer, axis = 1)
    test['visit_ts'] = test[['visitStartTime', '_timeZoneId']].apply(time_localizer, axis = 1)

    train = extract_time(train)
    test = extract_time(test)

    del train['_search_term']
    del test['_search_term']
    del train['visit_ts']
    del test['visit_ts']

    return train, test

def cat_conv(train, test):
    train['device_browser'] = train['device_browser'].map(lambda x: browser_mapping(str(x).lower())).astype('str')
    train['trafficSource_adContent'] = train['trafficSource_adContent'].map(
        lambda x: adcontents_mapping(str(x).lower())).astype('str')
    train['trafficSource_source'] = train['trafficSource_source'].map(lambda x: source_mapping(str(x).lower())).astype(
        'str')

    test['device_browser'] = test['device_browser'].map(lambda x: browser_mapping(str(x).lower())).astype('str')
    test['trafficSource_adContent'] = test['trafficSource_adContent'].map(
        lambda x: adcontents_mapping(str(x).lower())).astype('str')
    test['trafficSource_source'] = test['trafficSource_source'].map(lambda x: source_mapping(str(x).lower())).astype(
        'str')
    test['device_operatingSystem'] = test['device_operatingSystem'].map(lambda x: device_mapping(str(x).lower())).astype(
        'str')

    return train, test

def numeric_interaction_terms(df):
    df['totals_pageviews / totals_hits'] = df['totals_pageviews'] / df['totals_hits']
    df['visitNumber * totals_pageviews'] = df['visitNumber'] * df['totals_pageviews']
    df['visitNumber * totals_hits'] = df['visitNumber'] * df['totals_hits']
    df['visitNumber * totals_pageviews / totals_hits'] = df['visitNumber'] * df['totals_pageviews'] / df['totals_hits']

    return df

def drop_cols(train, test):
    drop_cols = ['socialEngagementType', 'trafficSource_campaign', 'trafficSource_adContent', 'trafficSource_keyword',
                 'trafficSource_campaign']
    train.drop(drop_cols, axis=1, inplace=True)
    test.drop(drop_cols, axis=1, inplace=True)

    return train, test



def categories_in_both(train, test):
    cats = ['device_browser', 'device_deviceCategory',
    'device_operatingSystem', 'geoNetwork_city', 'geoNetwork_continent',
    'geoNetwork_country', 'geoNetwork_metro', 'geoNetwork_networkDomain',
    'geoNetwork_region', 'geoNetwork_subContinent',
    'trafficSource_adContent', 'trafficSource_campaign',
    'trafficSource_keyword', 'trafficSource_medium',
    '_timeZoneId',
    'trafficSource_referralPath', 'trafficSource_source']
    for c in cats:
        print(c)
        train.loc[train[c].isnull(), c] = '(not set)'
        test.loc[test[c].isnull(), c] = '(not set)'
        intersect = np.intersect1d(train[c].unique(), test[c].unique())
        train.loc[~train[c].isin(intersect), c] = '(not set)'
        test.loc[~test[c].isin(intersect), c] = '(not set)'

    return train, test

def date(df):
    mean_cols = ['totals_hits', 'totals_pageviews']
    gr = df[mean_cols + ['date']].groupby(['date'])
    agg = gr.agg('mean')
    agg.columns = ['mean_' + x for x in agg.columns]
    df = df.merge(agg, left_on='date', right_index=True)

    sum_cols = ['totals_hits', 'totals_pageviews']
    gr = df[sum_cols + ['date']].groupby(['date'])
    agg = gr.agg('sum')
    agg.columns = ['sum_' + x for x in agg.columns]
    df = df.merge(agg, left_on='date', right_index=True)

    df['unix_hr'] = df['visitStartTime'] // 3600
    gr = df[mean_cols + ['unix_hr']].groupby(['unix_hr'])
    agg = gr.agg('mean')
    agg.columns = ['mean_unix_hr_' + x for x in agg.columns]
    df = df.merge(agg, left_on='unix_hr', right_index=True)

    gr = df[sum_cols + ['unix_hr']].groupby(['unix_hr'])
    agg = gr.agg('sum')
    agg.columns = ['sum_unix_hr_' + x for x in agg.columns]
    df = df.merge(agg, left_on='unix_hr', right_index=True)
    del df['unix_hr']

    gr = df[mean_cols + ['geoNetwork_country']].groupby(['geoNetwork_country'])
    agg = gr.agg('mean')
    agg.columns = ['mean_country_' + x for x in agg.columns]
    df = df.merge(agg, left_on='geoNetwork_country', right_index=True)

    gr = df[sum_cols + ['geoNetwork_country']].groupby(['geoNetwork_country'])
    agg = gr.agg('sum')
    agg.columns = ['sum_country_' + x for x in agg.columns]
    df = df.merge(agg, left_on='geoNetwork_country', right_index=True)


    return df

def target_encoding(train, test):
    encode_cols = ['geoNetwork_country', 'trafficSource_source', '_timeZoneId']
    for col in encode_cols:
        gr = train[['totals_transactionRevenue'] + [col]].groupby(col)
        agg = gr.agg('mean')
        agg.columns = ['target_enc_' + col]
        train = train.merge(agg, left_on=col, right_index=True)
        test = test.merge(agg, left_on=col, right_index=True)

    train.drop(encode_cols, axis=1, inplace=True)
    test.drop(encode_cols, axis=1, inplace=True)

    return train, test


def main(nrows=None):
    train = pd.read_pickle('../input/train_ext_json.pkl')
    test = pd.read_pickle('../input/test_ext_json.pkl')

    train['totals_transactionRevenue'] = train['totals_transactionRevenue'].fillna(0)
    if 'totals_transactionRevenue' in test.columns:
        del test['totals_transactionRevenue']
    train, test = timezone_conv(train, test)
    train, test = cat_conv(train, test)

    # train = process_device(train)
    # test = process_device(test)
    # train = geo(train)
    # test = geo(test)
    # train = traffic_source(train)
    # test = traffic_source(test)
    train = date(train)
    test = date(test)
    train, test = categories_in_both(train, test)
    train, test = target_encoding(train, test)
    train, test = drop_cols(train, test)

    # to_interact_cols = ['visitNumber', 'totals_hits', 'totals_pageviews']
    #
    train = numeric_interaction_terms(train)
    test = numeric_interaction_terms(test)

    train.to_pickle(os.path.join('..', 'input', 'train_fe.pkl'))
    test.to_pickle(os.path.join('..', 'input', 'test_fe.pkl'))

if __name__ == '__main__':
    main()
