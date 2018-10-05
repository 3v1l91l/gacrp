from sklearn.model_selection import GroupKFold, KFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns

# def get_folds(df=None, n_splits=5):
#     """Returns dataframe indices corresponding to Visitors Group KFold"""
#     # Get sorted unique visitors
#     unique_vis = np.array(sorted(df['fullVisitorId'].unique()))
#
#     folds = GroupKFold(n_splits=n_splits)
#     fold_ids = []
#     ids = np.arange(df.shape[0])
#     for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
#         fold_ids.append(
#             [
#                 ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
#                 ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
#             ]
#         )
#     # kf = KFold(n_splits=n_splits)
#     # fold_ids = kf.split((df))
#
#     return fold_ids

def get_folds(df):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    ix_sorted = df.sort_values('date').index.values
    train_to = int(len(ix_sorted) * 0.9)
    fold_ids = [(ix_sorted[:train_to], ix_sorted[train_to:])]
    return fold_ids

def main():
    train = pd.read_pickle(os.path.join('..', 'input', 'train_fe.pkl'))
    test = pd.read_pickle(os.path.join('..', 'input', 'test_fe.pkl'))
    encode_cols = ['geoNetwork_country', 'trafficSource_source', '_timeZoneId']
    train.drop(encode_cols, axis=1, inplace=True)

    ix_sorted = train.sort_values('date').index.values
    train_to = int(len(ix_sorted) * 0.9)
    fold_ids = [(ix_sorted[:train_to], ix_sorted[train_to:])]
    train['train'] = False
    train.loc[fold_ids[0][0], 'train'] = True

    excluded_features = [
        'date', 'fullVisitorId', 'sessionId', 'totals_transactionRevenue',
        'visitId', 'visitStartTime'
    ]

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
    n_splits = 1
    # folds = get_folds(train)
    folds = [(train[train['train'] == True].index.values, train[train['train'] == False].index.values)]

    importances = pd.DataFrame()
    importances['feature'] = train_features
    importances['gain'] = 0

    oof_reg_preds = np.zeros(train.shape[0])
    sub_reg_preds = np.zeros(test.shape[0])
    # for fold_, (trn_, val_) in enumerate(folds):
    #     print("Fold:", fold_)
    #     trn_x, trn_y = train[train_features].iloc[trn_], y_reg.iloc[trn_]
    #     val_x, val_y = train[train_features].iloc[val_], y_reg.iloc[val_]
    #     reg = lgb.LGBMRegressor(
    #         num_leaves=31,
    #         learning_rate=0.03,
    #         n_estimators=1000,
    #         subsample=.93,
    #         colsample_bytree=.94,
    #         random_state=1
    #     )
    #     reg.fit(
    #         trn_x, np.log1p(trn_y),
    #         eval_set=[(trn_x, np.log1p(trn_y)), (val_x, np.log1p(val_y))],
    #         eval_names=['TRAIN', 'VALID'],
    #         early_stopping_rounds=50,
    #         verbose=100,
    #         eval_metric='rmse',
    #         # categorical_feature=categorical_features
    #     )
    #     importances['gain'] += reg.booster_.feature_importance(importance_type='gain') / n_splits
    #
    #     oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    #     oof_reg_preds[oof_reg_preds < 0] = 0
    #     _preds = reg.predict(test[train_features], num_iteration=reg.best_iteration_)
    #     _preds[_preds < 0] = 0
    #     sub_reg_preds += np.expm1(_preds) / len(folds)

    (trn_, val_) = folds
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

    mse_session = mean_squared_error(np.log1p(y_reg[val_]), oof_reg_preds) ** .5
    print('mse_session {}'.format(mse_session))
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
    train = pd.read_pickle(os.path.join('..', 'input', 'train_session.pkl'))
    test = pd.read_pickle(os.path.join('..', 'input', 'test_session.pkl'))
    train_features += ['predictions']

    num_features = [_f for _f in train.columns if _f not in categorical_features + ['fullVisitorId', 'totals_transactionRevenue', 'predictions']]

    valid = train[train['train'] == False]
    train = train[train['train'] == True]
    train_cat = train[categorical_features + ['fullVisitorId']].groupby('fullVisitorId').first()
    train_pred = train[['fullVisitorId', 'predictions']].groupby('fullVisitorId').sum()
    train_num = train[num_features + ['fullVisitorId']].groupby('fullVisitorId').mean()
    y_reg_train = train['totals_transactionRevenue']

    valid_cat = valid[categorical_features + ['fullVisitorId']].groupby('fullVisitorId').first()
    valid_pred = valid[['fullVisitorId', 'predictions']].groupby('fullVisitorId').sum()
    valid_num = valid[num_features + ['fullVisitorId']].groupby('fullVisitorId').mean()
    y_reg_valid = valid['totals_transactionRevenue']

    train_data = pd.concat([train_pred, train_num, train_cat], axis=1)
    valid_data = pd.concat([valid_pred, valid_num, valid_cat], axis=1)
    del train_pred, train_num, train_cat
    del valid_pred, valid_num, valid_cat
    gc.collect()
    folds =1
    test_cat = test[categorical_features + ['fullVisitorId']].groupby('fullVisitorId').first()
    test_pred = test[['fullVisitorId', 'predictions']].groupby('fullVisitorId').sum()
    test_num = test[[x for x in num_features if x in test.columns] + ['fullVisitorId']].groupby('fullVisitorId').mean()

    sub_full_data = pd.concat([test_pred, test_num, test_cat], axis=1)
    del test_pred, test_num, test_cat
    gc.collect()

    trn_user_target = train[['fullVisitorId', 'target']].groupby('fullVisitorId').sum()
    n_splits = 1
    # folds = get_folds(df=full_data[['totals_pageviews']].reset_index(), n_splits=n_splits)

    oof_preds = np.zeros(valid_data.shape[0])
    sub_preds = np.zeros(sub_full_data.shape[0])
    importances = pd.DataFrame()
    importances['feature'] = train_features
    importances['gain'] = 0

    trn_x, trn_y = train_data[train_features], y_reg_train
    val_x, val_y = valid_data[train_features], y_reg_valid

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
        # categorical_feature=categorical_features,
        verbose=100
    )

    importances['gain'] += reg.booster_.feature_importance(importance_type='gain') / n_splits

    oof_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_preds[oof_preds < 0] = 0

    # Make sure features are in the same order
    _preds = reg.predict(sub_full_data[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_preds += _preds / len(folds)

    pd.DataFrame(data={'pred': oof_preds}).to_csv('pred.csv')
    oof_preds[oof_preds < np.percentile(oof_preds, 93)] = 0
    sub_preds[sub_preds < np.percentile(sub_preds, 93)] = 0
    mse_user =  mean_squared_error(np.log1p(y_reg_valid), oof_preds) ** .5
    print('mse_user {}'.format(mse_user))

    importances.to_csv('importance_user.csv')
    plt.figure(figsize=(20, 25))
    sns.barplot(x='gain', y='feature', data=importances.sort_values('gain', ascending=False).iloc[:300])
    plt.savefig('importances_user' + '.png')

    sub_full_data['PredictedLogRevenue'] = sub_preds
    sub_full_data[['PredictedLogRevenue']].to_csv('sub.csv', index=True)

if __name__ == '__main__':
    main()
