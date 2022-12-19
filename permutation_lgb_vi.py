# """
# permutation importance
# """

# from IPython.display import display
# from eli5.permutation_importance import get_score_importances
# from eli5.sklearn import PermutationImportance
# import eli5

# params =  {
#     'learning_rate': 0.07398,
#     'max_depth': 4.309,
#     'colsample_bytree': 0.4028,
#     'subsample': 0.4278,
#     'min_child_samples': 25.65,
#     'min_child_weight': 0.6138,
#     'min_split_gain': 0.7354,
#     'num_leaves': 62.68,
#     'reg_alpha': 0.2889,
#     'reg_lambda': 7.875,

# }

# permutation_importance_df = permutation_lgb_vi(train, params, drop_features=[],TARGET=TARGET)

# # save
# permutation_importance_df.to_csv('permutation_importance_df.csv',index=False)

# # check 
# a1 = (permutation_importance_df.loc[permutation_importance_df['weight']<0,'feature'].value_counts()>=4)
# drop_features_vc4 = a1.index[a1].tolist()
# print('drop_features_vc4:',len(drop_features_vc4))

# a1 = (permutation_importance_df.loc[permutation_importance_df['weight']<0,'feature'].value_counts()>=3)
# drop_features_vc3 = a1.index[a1].tolist()
# print('drop_features_vc3:',len(drop_features_vc3))

# a1 = (permutation_importance_df.loc[permutation_importance_df['weight']<0,'feature'].value_counts()>=2)
# drop_features_vc2 = a1.index[a1].tolist()
# print('drop_features_vc2:',len(drop_features_vc2))

# # check 
# print('drop_features_vc2=',drop_features_vc2)
# print('drop_features_vc3=',drop_features_vc3)
# print('drop_features_vc4=',drop_features_vc4)

# # set drop_features
# drop_features = drop_features_vc2


def permutation_lgb_vi(train, params, drop_features,TARGET):
    
    params['objective'] = 'multiclass'
    params['learning_rate'] = max(min(params['learning_rate'], 1), 0)
    params["num_leaves"] = int(round(params['num_leaves']))
    params['colsample_bytree'] = max(min(params['colsample_bytree'], 1), 0)
    params['subsample'] = max(min(params['subsample'], 1), 0)
    params['max_depth'] = int(round(params['max_depth']))        
    params['reg_alpha'] = params['reg_alpha']
    params['reg_lambda'] = params['reg_lambda']        
    params['min_split_gain'] = params['min_split_gain']
    params['min_child_weight'] = params['min_child_weight']
    params['min_child_samples'] = int(round(params['min_child_samples']))

    train_df = train.copy()

    # ----
    
    # label encoder 
    class_le = preprocessing.LabelEncoder()
    snp_le = preprocessing.LabelEncoder()
    snp_col = [f'SNP_{str(x).zfill(2)}' for x in range(1,16)]

    snp_data = []
    for col in snp_col:
        snp_data += list(train_df[col].values)

    # encoding - y 
    train_df[TARGET] = class_le.fit_transform(train_df[TARGET])
    snp_le.fit(snp_data)

    # encoding - x
    for col in train_df.columns:
        if col in snp_col:
            train_df[col] = snp_le.transform(train_df[col])           
    # ----

    stratified = True

    if stratified:
        folds = StratifiedKFold(n_splits= 5, shuffle=True, random_state=1)
    else:
        folds = KFold(n_splits= 5, shuffle=True, random_state=1)

    FEATURES = [f for f in train_df.columns if f not in [TARGET,'id']+drop_features]
    TARGET_COL = TARGET
    feature_importance_df = pd.DataFrame()
    for fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_df[TARGET_COL])):
        clf = LGBMClassifier(
            **params, 
            random_state=42, 
            n_jobs = -1,
            silent=True,
            n_estimators=1000
        )

        with warnings.catch_warnings():

            warnings.filterwarnings('ignore')

            clf.fit(
                train_df.loc[train_idx, FEATURES], 
                train_df.loc[train_idx, TARGET_COL], 
                eval_metric="multi_logloss", 
                verbose=False,
                early_stopping_rounds=500,
                eval_set=[(train_df.loc[valid_idx, FEATURES],train_df.loc[valid_idx, TARGET_COL])]
            )    
            
        # scoring = 'f1'
        perm = PermutationImportance(clf, n_iter=100, scoring='accuracy',random_state=1).fit(train_df.loc[valid_idx, FEATURES],train_df.loc[valid_idx, TARGET_COL])
        fold_importance_df = eli5.explain_weights_df(perm, top = len(FEATURES), feature_names = FEATURES)
        fold_importance_df["fold"] = fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print(f"Permutation importance for fold {fold}")
    
    return feature_importance_df
