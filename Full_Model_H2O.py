import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
import h2o
h2o.init()
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator




flag_early = 0
if flag_early:
    tag_pred = 'early'
else:
    tag_pred = 'late'




# LOAD AND MERGE DATA FILES

# load
path = '/data/WorkData/spatialtemporal/'
file = pd.read_csv(path + 'finalmerge_any_' + tag_pred + 'v2.csv')
timefile = pd.read_csv(path + 'gr_lastyear_' + tag_pred + '.csv')
test_cases = pd.read_csv(path + 'test_cases_last_hearingv2.csv', header=None)

# merge
file = pd.merge(file, timefile, on=['idnproceeding','idncase'], how='left')


# In[5]:


test_cases = test_cases.rename(columns={0:'num'})
train = file[~file.idncase.isin(test_cases.num)]
test = file[file.idncase.isin(test_cases.num)]




# columns to drop in transformer depending on the model
if flag_early: 
    cols = ['idncase', 'idnproceeding','adj_date','osc_date','base_city_code', 'hearing_loc_code', 
	'notice_desc','adj_time_start2','adj_time_stop2', 'numAppsPerProc','numProcPerCase','adj_rsn_desc']
else: 
    cols = ['idncase', 'idnproceeding','adj_date','comp_date','osc_date','numAppsPerProc','numProcPerCase','base_city_code',
            'hearing_loc_code', 'notice_desc','adj_time_start2','adj_time_stop2','adj_rsn_desc']



def transform(X):
    # drop unused columns        
    X = X.drop(columns=cols)
    # save list of variables
    training = X.columns.values.tolist()[1:]
    response = 'dec'
    # changes frame to H2O compatible frame
    X = h2o.H2OFrame(X)
    # encodes as categorical
    X['dec'] = X['dec'].asfactor()
    return X, training, response



#clean = Cleaning()
train, training_columns, response_column = transform(train)


model = H2ORandomForestEstimator(ntrees=100, max_depth=20, nfolds=5, keep_cross_validation_predictions= True, stopping_metric='auc')
model.train(x=training_columns, y=response_column, training_frame=train)


print(model)

