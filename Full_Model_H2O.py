def rf_models(path,flag_baseline,flag_early,flag_spatialonly,flag_temponly):
     # ## Runs logistic regression models
    # FLAGS TO TOGGLE BETWEEN DIFFERENT MODELS
    # flag_baseline: 
    #   1 - absence of any spatial or temporal features (should be with flag_early = 1)
    #   0 - all features

    # flag_early: 
    #   1 - early predictability. features a function of osc_date. 
    #   0 - late predictability. features a function of last hearing.
    # flag_spatialonly: baseline features + spatial features
    # flag_temponly: baseline features + temporal features
    # ======= PACKAGES, FILEPATHS, FLAGS ========
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.base import BaseEstimator, TransformerMixin
    import h2o
    h2o.init()
    from h2o.estimators.gbm import H2OGradientBoostingEstimator
    from h2o.estimators.random_forest import H2ORandomForestEstimator
    from h2o.grid.grid_search import H2OGridSearch
    import matplotlib.pyplot as plt
    import pickle

     # used for loading appropriate dataset file
    if flag_early:
        tag_pred = 'early'
    else:
        tag_pred = 'late'


    # LOAD AND MERGE DATA FILES

    # load
    file = pd.read_csv(path + '/finalmerge_any_' + tag_pred + '_final.csv')
    timefile = pd.read_csv(path + '/gr_lastyear_' + tag_pred + '.csv')
    test_cases = pd.read_csv(path + '/test_cases_last_hearing_final.csv', header=None)

    # merge
    if (not flag_baseline) and (not flag_spatialonly): file = pd.merge(file, timefile, on=['idnproceeding','idncase'], how='left')

    # get training dataset
    test_cases = test_cases.rename(columns={0:'num'})
    train = file[~file.idncase.isin(test_cases.num)]
    test = file[file.idncase.isin(test_cases.num)]

    # ====== GET RELEVANT FEATURES FOR EACH MODEL =======
    # features to remove
    if flag_early: # early predictability
        cols = ['idncase', 'idnproceeding','adj_date','osc_date','base_city_code', 'hearing_loc_code', 
    	'notice_desc','adj_time_start2','adj_time_stop2', 'numAppsPerProc','numProcPerCase','adj_rsn_desc']
    else: # late predictability
        cols = ['idncase', 'idnproceeding','adj_date','comp_date','osc_date','numAppsPerProc','numProcPerCase','base_city_code',
                'hearing_loc_code', 'notice_desc','adj_time_start2','adj_time_stop2','adj_rsn_desc']

    if flag_spatialonly: # spatial features only
        if flag_early: cols2 = ['osc_date_delta','pres_aff','hearingYear','hearingMonth','hearingDayOfWeek']
        else: cols2 = ['osc_date_delta','pres_aff','hearingYear','hearingMonth','hearingDayOfWeek',
                      'numHearingsPerProc','durationFirstLastHearing','caseDuration']

    if flag_temponly: # temporal features only
        cols2 = ['hearing_loc_code','base_city','hearing_city']

    if flag_temponly: # temporal features only
        cols2 = ['hearing_loc_code','base_city','hearing_city']

    # features to include
    if flag_baseline: # baseline model only
        cols = ['dec','tracid','nat','c_asy_type','case_type','lang_hearing',
                'attorney_flag','sched_type','adj_medium'] # dropping spatial and temporal variables

    def transform(X):
        # drop unused columns        
        if flag_baseline: X = X[cols]
        else: X = X.drop(columns=cols)

        if flag_spatialonly: X = X.drop(columns=cols2)
        if flag_temponly: X = X.drop(columns=cols2)
        # save list of variables
        training = X.columns.values.tolist()[1:]
        response = 'dec'
        # changes frame to H2O compatible frame
        X = h2o.H2OFrame(X)
        # encodes as categorical
        X['dec'] = X['dec'].asfactor()
        X['tracid'] = X['tracid'].asfactor()
        X['attorney_flag'] = X['attorney_flag'].asfactor()
        return X, training, response



    train, training_columns, response_column = transform(train)

    #model = H2ORandomForestEstimator(ntrees=100, max_depth=20, nfolds=5, keep_cross_validation_predictions= True, stopping_metric='auc')
    #model.train(x=training_columns, y=response_column, training_frame=train)
    #print(model)


    # hyperparameter searching
    estimator = H2ORandomForestEstimator(stopping_tolerance = 0.001, stopping_metric = 'auc', nfolds=5,  seed = 44)

    hyper_parameters = {'ntrees':[50, 100, 250, 500], 
                        'max_depth':[2, 5, 10]}

    grid_search = H2OGridSearch(model = estimator, 
                                hyper_params = hyper_parameters)

    grid_search.train(x = training_columns,
                      y = response_column,
                      training_frame = train)

    results = grid_search.get_grid(sort_by='auc',decreasing=True)

    best_model = results.models[0]


    #  plotting and saving feature importances
    plt.rcdefaults()
    fig, ax = plt.subplots()
    variables = best_model._model_json['output']['variable_importances']['variable']
    y_pos = np.arange(len(variables))
    scaled_importance = best_model._model_json['output']['variable_importances']['scaled_importance']
    ax.barh(y_pos, scaled_importance)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.invert_yaxis()
    ax.set_xlabel('Scaled Importance')
    ax.set_title('Variable Importance')
    plt.tight_layout()

    #model names    
    if flag_baseline:
        name = 'baseline'
    elif flag_early:
        if flag_spatialonly:
            name = 'early_spatial'
        elif flag_temponly:
            name = 'early_temporal'
        else:
            name = 'early'
    else:
        if flag_spatialonly:
            name = 'late_spatial'
        elif flag_temponly:
            name = 'late_temporal'
        else:
            name = 'late'

    # saving best model summary and feature weights
    with open(path + '/rf_' + name + '_best_model_summary.csv', 'w') as f:
        best_model._model_json['output']['model_summary'].as_data_frame().to_csv(f, header=True, index=False)
        f.close()
    with open(path + '/rf_' + name + '_best_model_features.csv', 'w') as f:
        best_model._model_json['output']['variable_importances'].as_data_frame().to_csv(f, header=True, index=False)
        f.close()

    # saving feature importance plot
    plt.savefig(path + '/rf_' + name + '_feature_importance.png')
    plt.clf()

    # save H2O model obect 
    new_model_path = h2o.save_model(best_model, path + '/rf_' + name + '_best_model',  force=True)

    print('Model: '+name)
    print('AUC: '+str(test_result.auc()))
    print(results)
     # testing on test set
    #test, training_columns, response_column = transform(test)
    #test_result = best_model.model_performance(test)
    #print('Evaluating best model on test set')
    #print('AUC: '+str(test_result.auc()))

