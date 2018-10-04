
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
    data_df['source_country'] = data_df['trafficSource_source'] + '_' + data_df['geoNetwork_country']
    data_df['campaign_medium'] = data_df['trafficSource_campaign'] + '_' + data_df['trafficSource_medium']
    data_df['browser_category'] = data_df['device_browser'] + '_' + data_df['device_deviceCategory']
    data_df['browser_os'] = data_df['device_browser'] + '_' + data_df['device_operatingSystem']
    return data_df

def custom(data):
    print('custom..')
    data['device_deviceCategory_channelGrouping'] = data['device_deviceCategory'] + "_" + data['channelGrouping']
    data['channelGrouping_browser'] = data['device_browser'] + "_" + data['channelGrouping']
    data['channelGrouping_OS'] = data['device_operatingSystem'] + "_" + data['channelGrouping']

    for i in ['geoNetwork_city', 'geoNetwork_continent', 'geoNetwork_country', 'geoNetwork_metro',
              'geoNetwork_networkDomain', 'geoNetwork_region', 'geoNetwork_subContinent']:
        for j in ['device_browser', 'device_deviceCategory', 'device_operatingSystem', 'trafficSource_source']:
            data[i + "_" + j] = data[i] + "_" + data[j]

    data['content_source'] = data['trafficSource_adContent'] + "_" + data['source_country']
    data['medium_source'] = data['trafficSource_medium'] + "_" + data['source_country']
    return data

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

    train['date'] = train[['visitStartTime', '_timeZoneId']].apply(time_localizer, axis = 1)
    test['date'] = test[['visitStartTime', '_timeZoneId']].apply(time_localizer, axis = 1)
    for df in [train, test]:
        df['sess_date_dow'] = df['date'].map(lambda x: x.weekday()).astype('int8')
        df['sess_date_hours'] = df['date'].map(lambda x: x.strftime('%H')).astype('int8')

    del train['_search_term']
    del test['_search_term']

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

    train = process_device(train)
    test = process_device(test)
    train = custom(train)
    test = custom(test)

    return train, test

def numeric_interaction_terms(df, columns):
    for c in combinations(columns, 2):
        df['{} / {}'.format(c[0], c[1])] = df[c[0]] / df[c[1]]
        df['{} * {}'.format(c[0], c[1])] = df[c[0]] * df[c[1]]
        df['{} - {}'.format(c[0], c[1])] = df[c[0]] - df[c[1]]
    return df

def main(nrows=None):
    train = pd.read_pickle('../input/train_ext_json.pkl')
    test = pd.read_pickle('../input/test_ext_json.pkl')

    train['totals_transactionRevenue'] = train['totals_transactionRevenue'].fillna(0)
    if 'totals_transactionRevenue' in test.columns:
        del test['totals_transactionRevenue']

    train, test = timezone_conv(train, test)
    # train, test = cat_conv(train, test)

    to_interact_cols = ['visitNumber', 'totals_hits', 'totals_pageviews']

    train = numeric_interaction_terms(train, to_interact_cols)
    test = numeric_interaction_terms(test, to_interact_cols)

    train.to_pickle(os.path.join('..', 'input', 'train_fe.pkl'))
    test.to_pickle(os.path.join('..', 'input', 'test_fe.pkl'))

if __name__ == '__main__':
    main()
