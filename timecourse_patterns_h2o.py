#fits models described in "recency analysis" section of the paper, plots weights of features and fits exponential function to those weights

def timecourse_pattern_analysis(path):
    from fit_exp_plots import exp_decay_models
    import h2o
    h2o.init()
    def run_model(h2o,path,f_type):
        #f_type ='judge' or 'nat'
        import pandas as pd
        import numpy as np

        from h2o.estimators.random_forest import H2ORandomForestEstimator

       
        # this is to toggle the 2 types of time course variables
        if f_type=='judge':
            flag_tc = 1
        elif f_type=='nat':
            flag_tc=0

        if flag_tc:   
            tc_cols = ['tracid_gr_1_10d','tracid_gr_11_20d',
                   'tracid_gr_21_30d', 'tracid_gr_31_40d', 'tracid_gr_41_50d',
                    'tracid_gr_51_60d', 'tracid_gr_61_70d',
                   'tracid_gr_71_80d', 'tracid_gr_81_90d','tracid_gr_91_100d']
        else:
            tc_cols = ['nat_gr_1yr', 'nat_gr_2yr','nat_gr_3yr','nat_gr_4yr','nat_gr_5yr',
                        'nat_gr_6yr','nat_gr_7yr','nat_gr_8yr','nat_gr_9yr','nat_gr_10yr']

        # loading appropriate files
        file = pd.read_csv(path + '/finalmerge_any_late_final.csv')
        timefile = pd.read_csv(path + '/gr_lastyear_late.csv')
        extra_timefile = pd.read_csv(path + '/extra_timefeatures_late.csv')
        test_cases = pd.read_csv(path + '/test_cases_last_hearing_final.csv', header=None)

        #merge timefeatures with other features
        file = pd.merge(file, timefile, on=['idnproceeding','idncase'])
        file = pd.merge(file, pd.concat((extra_timefile['idnproceeding'],extra_timefile[tc_cols]),axis=1), on='idnproceeding')

        # get training data set
        test_cases = test_cases.rename(columns={0:'num'})
        train = file[~file.idncase.isin(test_cases.num)]
        #simplify model--only a few features beyond the timecourse ones
        cols =  ['dec','tracid','nat','attorney_flag','c_asy_type','case_type']+tc_cols
        train_data = train[cols]

        
        #convert to h2o dataframe
        train_data = h2o.H2OFrame(train_data)
        
        
        # change variables that are incorrect dtypes
        train_data['attorney_flag'] = train_data['attorney_flag'].asfactor()
        train_data['tracid'] = train_data['tracid'].asfactor()
        train_data['dec'] = train_data['dec'].asfactor()
        training_columns = train_data.columns[1:]
        response_column = 'dec'
        
        #run model
        model = H2ORandomForestEstimator(ntrees=100, max_depth=20, nfolds=5, 
                                         keep_cross_validation_predictions= True, stopping_metric='auc')

        model.train(x=training_columns, y=response_column, training_frame=train_data)

        #  weights --just time features
        scaled_importance  = model._model_json['output']['variable_importances']['scaled_importance']
        variables = model._model_json['output']['variable_importances']['variable']
        indices = [i for i in range(len(variables)) if variables[i] in tc_cols]
        
        
        df = pd.DataFrame.from_dict({'feature':[variables[i] for i in indices],
                                     'weight':[scaled_importance[i] for i in indices]})
  
        df.to_csv(path + '/rf_' +  f_type +  '_exp_weights_clean.csv')
        return([scaled_importance[i] for i in indices])
    judge_weights = run_model(h2o,path,'judge')
    nat_weights = run_model(h2o,path,'nat')
    exp_decay_models(path,nat_weights,judge_weights)
    
        
        
                                     

