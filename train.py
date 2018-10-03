from sklearn.model_selection import GroupKFold
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

    # Get folds
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

    return fold_ids

def main():
    train = pd.read_csv(os.path.join('..', 'input', 'train.gz'),
                        dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
    test = pd.read_csv(os.path.join('..', 'input', 'test.gz'),
                       dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)

    excluded_features = [
        'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue',
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

    y_reg = train['totals.transactionRevenue']
    n_splits = 5
    folds = get_folds(df=train, n_splits=n_splits)

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
            num_leaves=20,
            max_depth=7,
            learning_rate=0.03,
            n_estimators=1000,
            subsample=.93,
            colsample_bytree=.94,
            random_state=1
        )
        reg.fit(
            trn_x, np.log1p(trn_y),
            eval_set=[(val_x, np.log1p(val_y))],
            early_stopping_rounds=50,
            verbose=100,
            eval_metric='rmse',
            categorical_feature=categorical_features
        )
        importances['gain'] += reg.booster_.feature_importance(importance_type='gain') / n_splits

        oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
        oof_reg_preds[oof_reg_preds < 0] = 0
        _preds = reg.predict(test[train_features], num_iteration=reg.best_iteration_)
        _preds[_preds < 0] = 0
        sub_reg_preds += np.expm1(_preds) / len(folds)

    mse_session = mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5
    print('mse_user {}'.format(mse_session))
    train['predictions'] = np.expm1(oof_reg_preds)
    test['predictions'] = sub_reg_preds

    train.to_pickle(os.path.join('..', 'input', 'train_session.pkl'))
    test.to_pickle(os.path.join('..', 'input', 'test_session.pkl'))

    importances.to_csv('importances_session.csv')
    plt.figure(figsize=(8, 12))
    sns.barplot(x='gain', y='feature', data=importances.sort_values('gain', ascending=False))
    plt.savefig('importances_session' + '.png')

def user_level(train_features, categorical_features):
    train = pd.read_pickle(os.path.join('..', 'input', 'train_session.pkl'))
    test = pd.read_pickle(os.path.join('..', 'input', 'test_session.pkl'))

    y_reg = train['totals.transactionRevenue']

    num_features = [_f for _f in train.columns if _f not in categorical_features + ['fullVisitorId', 'totals.transactionRevenue', 'predictions']]

    train_cat = train[categorical_features + ['fullVisitorId']].groupby('fullVisitorId').first()
    train_pred = train[['fullVisitorId', 'predictions']].groupby('fullVisitorId').sum()
    train_num = train[num_features + ['fullVisitorId']].groupby('fullVisitorId').mean()

    # Create a DataFrame with VisitorId as index
    # trn_pred_list contains dict
    # so creating a dataframe from it will expand dict values into columns
    # trn_all_predictions = pd.DataFrame(list(trn_pred_list.values), index=trn_data.index)
    # trn_feats = trn_all_predictions.columns
    # trn_all_predictions['t_mean'] = np.log1p(trn_all_predictions[trn_feats].mean(axis=1))
    # trn_all_predictions['t_median'] = np.log1p(trn_all_predictions[trn_feats].median(axis=1))
    # trn_all_predictions['t_sum_log'] = np.log1p(trn_all_predictions[trn_feats]).sum(axis=1)
    # trn_all_predictions['t_sum_act'] = np.log1p(trn_all_predictions[trn_feats].fillna(0).sum(axis=1))
    # trn_all_predictions['t_nb_sess'] = trn_all_predictions[trn_feats].isnull().sum(axis=1)
    full_data = pd.concat([train_pred, train_num, train_cat], axis=1)
    del train_pred, train_num, train_cat
    gc.collect()


    test_cat = test[categorical_features + ['fullVisitorId']].groupby('fullVisitorId').first()
    test_pred = test[['fullVisitorId', 'predictions']].groupby('fullVisitorId').sum()
    test_num = test[[x for x in num_features if x in test.columns] + ['fullVisitorId']].groupby('fullVisitorId').mean()

    # sub_all_predictions = test[['fullVisitorId', 'predictions']].groupby('fullVisitorId').sum() \
    #     .apply(lambda df: list(df.predictions)) \
    #     .apply(lambda x: {'pred_' + str(i): pred for i, pred in enumerate(x)})
    #
    # sub_data = test[categorical_features + ['fullVisitorId']].groupby('fullVisitorId').first()
    # sub_all_predictions = pd.DataFrame(list(sub_pred_list.values), index=sub_data.index)
    # for f in trn_feats:
    #     if f not in sub_all_predictions.columns:
    #         sub_all_predictions[f] = np.nan
    # sub_all_predictions['t_mean'] = np.log1p(sub_all_predictions[trn_feats].mean(axis=1))
    # sub_all_predictions['t_median'] = np.log1p(sub_all_predictions[trn_feats].median(axis=1))
    # sub_all_predictions['t_sum_log'] = np.log1p(sub_all_predictions[trn_feats]).sum(axis=1)
    # sub_all_predictions['t_sum_act'] = np.log1p(sub_all_predictions[trn_feats].fillna(0).sum(axis=1))
    # sub_all_predictions['t_nb_sess'] = sub_all_predictions[trn_feats].isnull().sum(axis=1)
    # sub_full_data = pd.concat([sub_data, sub_all_predictions], axis=1)
    sub_full_data = pd.concat([test_pred, test_num, test_cat], axis=1)
    del test_pred, test_num, test_cat
    gc.collect()

    train['target'] = y_reg
    trn_user_target = train[['fullVisitorId', 'target']].groupby('fullVisitorId').sum()
    n_splits = 5
    folds = get_folds(df=full_data[['totals.pageviews']].reset_index(), n_splits=n_splits)

    oof_preds = np.zeros(full_data.shape[0])
    sub_preds = np.zeros(sub_full_data.shape[0])
    importances = pd.DataFrame()
    importances['feature'] = train_features
    importances['gain'] = 0

    for fold_, (trn_, val_) in enumerate(folds):
        trn_x, trn_y = full_data[train_features].iloc[trn_], trn_user_target['target'].iloc[trn_]
        val_x, val_y = full_data[train_features].iloc[val_], trn_user_target['target'].iloc[val_]

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
            verbose=100
        )

        importances['gain'] = reg.booster_.feature_importance(importance_type='gain') / n_splits

        oof_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
        oof_preds[oof_preds < 0] = 0

        # Make sure features are in the same order
        _preds = reg.predict(sub_full_data[train_features], num_iteration=reg.best_iteration_)
        _preds[_preds < 0] = 0
        sub_preds += _preds / len(folds)

    mse_user =  mean_squared_error(np.log1p(trn_user_target['target']), oof_preds) ** .5
    print('mse_user {}'.format(mse_user))

    importances.to_csv('importance_user.csv')
    plt.figure(figsize=(8, 25))
    sns.barplot(x='gain', y='feature', data=importances.sort_values('gain', ascending=False).iloc[:300])
    plt.savefig('importances_user' + '.png')

    sub_full_data['PredictedLogRevenue'] = sub_preds
    sub_full_data[['PredictedLogRevenue']].to_csv('new_test.csv', index=True)

if __name__ == '__main__':
    main()
