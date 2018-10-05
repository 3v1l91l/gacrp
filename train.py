from sklearn.model_selection import GroupKFold, KFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns

def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )
    # kf = KFold(n_splits=n_splits)
    # fold_ids = kf.split((df))

    return fold_ids

def main():
    train = pd.read_pickle(os.path.join('..', 'input', 'train_fe.pkl'))
    test = pd.read_pickle(os.path.join('..', 'input', 'test_fe.pkl'))

    excluded_features = [
        'date', 'fullVisitorId', 'sessionId', 'totals_transactionRevenue',
        'visitId', 'visitStartTime'
    ]
    # categorical_features = ['channelGrouping', 'device_browser', 'device_deviceCategory', 'device_operatingSystem', 'geoNetwork_city', 'geoNetwork_continent', 'geoNetwork_metro', 'geoNetwork_networkDomain', 'geoNetwork_region', 'geoNetwork_subContinent', 'trafficSource_medium', 'trafficSource_referralPath', 'ts_day_of_week']
    categorical_features = [
        _f for _f in train.columns
        if (_f not in excluded_features) & (train[_f].dtype == 'object')
    ]
    for f in categorical_features:
        train[f], indexer = pd.factorize(train[f])
        test[f] = indexer.get_indexer(test[f])

    train_features = [_f for _f in train.columns if _f not in excluded_features]
    # print(train_features)

    session_level(train, test, train_features, categorical_features)
    user_level(train_features, categorical_features)

def session_level(train, test, train_features, categorical_features):

    y_reg = train['totals_transactionRevenue']
    n_splits = 5
    folds = get_folds(train, n_splits)

    importances = pd.DataFrame()
    importances['feature'] = train_features
    importances['gain'] = 0

    oof_reg_preds = np.zeros(train.shape[0])
    sub_reg_preds = np.zeros(test.shape[0])
    for fold_, (trn_, val_) in enumerate(folds):
        print("Fold:", fold_)
        trn_x, trn_y = train[train_features].iloc[trn_], y_reg.iloc[trn_]
        val_x, val_y = train[train_features].iloc[val_], y_reg.iloc[val_]
        reg = lgb.LGBMRegressor(
            num_leaves=31,
            learning_rate=0.03,
            n_estimators=1000,
            subsample=.93,
            colsample_bytree=.94,
            random_state=1
        )
        reg.fit(
            trn_x, np.log1p(trn_y),
            eval_set=[(trn_x, np.log1p(trn_y)), (val_x, np.log1p(val_y))],
            eval_names=['TRAIN', 'VALID'],
            early_stopping_rounds=50,
            verbose=100,
            eval_metric='rmse',
            # categorical_feature=categorical_features
        )
        importances['gain'] += reg.booster_.feature_importance(importance_type='gain') / n_splits

        oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
        oof_reg_preds[oof_reg_preds < 0] = 0
        _preds = reg.predict(test[train_features], num_iteration=reg.best_iteration_)
        _preds[_preds < 0] = 0
        sub_reg_preds += np.expm1(_preds) / len(folds)

    mse_session = mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
    print('mse_session {}'.format(mse_session))
    ln =np.log1p(y_reg)
    mse_user_lb = mean_squared_error(ln[ln < 18.3], oof_reg_preds[ln < 18.3]) ** .5
    print('mse_session LB: {}'.format(mse_user_lb))

    pd.DataFrame(data={'pred': oof_reg_preds}).to_csv('sess_pred.csv')
    train['predictions'] = np.expm1(oof_reg_preds)
    test['predictions'] = sub_reg_preds

    train.to_pickle(os.path.join('..', 'input', 'train_session.pkl'))
    test.to_pickle(os.path.join('..', 'input', 'test_session.pkl'))

    importances.to_csv('importances_session.csv')
    plt.figure(figsize=(20, 18))
    sns.barplot(x='gain', y='feature', data=importances.sort_values('gain', ascending=False))
    plt.savefig('importances_session' + '.png')


def user_level(train_features, categorical_features):
    n_splits = 5

    train = pd.read_pickle(os.path.join('..', 'input', 'train_session.pkl'))
    test = pd.read_pickle(os.path.join('..', 'input', 'test_session.pkl'))

    y_reg = train['totals_transactionRevenue']

    num_features = [_f for _f in train.columns if
                    _f not in categorical_features + ['fullVisitorId', 'totals_transactionRevenue', 'predictions']]

    train_cat = train[categorical_features + ['fullVisitorId']].groupby('fullVisitorId').first()
    train_pred = train[['fullVisitorId', 'predictions']].groupby('fullVisitorId').sum()
    train_num = train[num_features + ['fullVisitorId']].groupby('fullVisitorId').mean()

    full_data = pd.concat([train_pred, train_num, train_cat], axis=1)
    del train_pred, train_num, train_cat
    gc.collect()

    test_cat = test[categorical_features + ['fullVisitorId']].groupby('fullVisitorId').first()
    test_pred = test[['fullVisitorId', 'predictions']].groupby('fullVisitorId').sum()
    test_num = test[[x for x in num_features if x in test.columns] + ['fullVisitorId']].groupby('fullVisitorId').mean()

    sub_full_data = pd.concat([test_pred, test_num, test_cat], axis=1)
    del test_pred, test_num, test_cat
    gc.collect()

    train['target'] = y_reg
    trn_user_target = train[['fullVisitorId', 'target']].groupby('fullVisitorId').sum()
    kf = KFold(n_splits=n_splits, random_state=1, shuffle=True)
    oof_preds = np.zeros(full_data.shape[0])
    sub_preds = np.zeros(sub_full_data.shape[0])
    importances = pd.DataFrame()
    importances['feature'] = train_features
    importances['gain'] = 0

    for (trn_, val_) in kf.split(full_data):
        trn_x, trn_y = full_data[train_features].iloc[trn_], trn_user_target['target'][trn_]
        val_x, val_y = full_data[train_features].iloc[val_], trn_user_target['target'][val_]

        reg = lgb.LGBMRegressor(
            num_leaves=31,
            learning_rate=0.03,
            n_estimators=1000,
            subsample=.9,
            colsample_bytree=.9,
            random_state=1
        )
        reg.fit(
            trn_x, np.log1p(trn_y),
            eval_set=[(trn_x, np.log1p(trn_y)), (val_x, np.log1p(val_y))],
            eval_names=['TRAIN', 'VALID'],
            early_stopping_rounds=50,
            eval_metric='rmse',
            verbose=100,
            categorical_feature=categorical_features
        )

        importances['gain'] = reg.booster_.feature_importance(importance_type='gain') / n_splits

        oof_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
        oof_preds[oof_preds < 0] = 0

        # Make sure features are in the same order
        _preds = reg.predict(sub_full_data[train_features], num_iteration=reg.best_iteration_)
        _preds[_preds < 0] = 0
        sub_preds += _preds / n_splits

    mse_user = mean_squared_error(np.log1p(trn_user_target['target']), oof_preds) ** .5
    print('mse_user {}'.format(mse_user))
    ln =np.log1p(trn_user_target['target'])
    mse_user_lb = mean_squared_error(ln[ln < 18.3], oof_preds[ln < 18.3]) ** .5
    print('mse_user LB: {}'.format(mse_user_lb))

    importances.to_csv('importance_user.csv')
    plt.figure(figsize=(20, 25))
    sns.barplot(x='gain', y='feature', data=importances.sort_values('gain', ascending=False).iloc[:300])
    plt.savefig('importances_user' + '.png')

    sub_full_data['PredictedLogRevenue'] = sub_preds
    sub_full_data[['PredictedLogRevenue']].to_csv('new_test.csv', index=True)

if __name__ == '__main__':
    main()
