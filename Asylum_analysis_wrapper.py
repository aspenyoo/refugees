from Cleaning_asylum import clean_master_app
from Cleaning_detailed_schedule import clean_detailed_schedule
from Cleaning_schedule import clean_schedule
from time_features import make_timefeatures
from Train_test_split import train_test_split
from Log_Reg_Models import log_reg_models
from Full_Model_H2O import rf_models
from timecourse_patterns_h2o import timecourse_pattern_analysis
import pandas as pd
import numpy as np
import random

pd.set_option('precision', 5)

#SEED THE RANDOM NUMBER GENERATOR
random.seed(44)

#set data paths
raw_path = '/data/Dropbox/Data/Asylum_Courts/raw'
analysis_path = '/data/WorkData/spatialtemporal'
tbl_schedule_path = '/data/Dropbox/Data/Asylum_Courts/tbl_schedule' 

print('cleaning master and app files')
#clean and merge master and application file
clean_master_app(raw_path,analysis_path,0) 

print('cleaning detailed schedule')
#clean detailed schedule file
clean_detailed_schedule(raw_path,tbl_schedule_path,analysis_path) 

#clean detailed schedule file, early and late versions
print('cleaning schedule: late') 
clean_schedule(raw_path,analysis_path,0,0)
print('cleaning schedule: early')
clean_schedule(raw_path,analysis_path,1,0)

#create grant history over time features
print('creating grant history features')
make_timefeatures(analysis_path) 

#split the data into training and test
print('splitting data into training and test')
train_test_split(analysis_path)


#run logistic regression models
print('running baseline model')
log_reg_models(analysis_path,1,1,0,0)
print('running early spatial')
log_reg_models(analysis_path,0,1,1,0)
print('running early temporal')
log_reg_models(analysis_path,0,1,0,1)
print('running early full')
log_reg_models(analysis_path,0,1,0,0)
print('running late spatial')
log_reg_models(analysis_path,0,0,1,0)
print('running late temporal')
log_reg_models(analysis_path,0,0,0,1)
print('running late full')
log_reg_models(analysis_path,0,0,0,0)


#run h2o models 
print('running baseline model')
rf_models(analysis_path,1,1,0,0)
print('running early spatial')
rf_models(analysis_path,0,1,1,0)
print('running early temporal')
rf_models(analysis_path,0,1,0,1)
print('running early full')
rf_models(analysis_path,0,1,0,0)
print('running late spatial')
rf_models(analysis_path,0,0,1,0)
print('running late temporal')
rf_models(analysis_path,0,0,0,1)
print('running late full')
rf_models(analysis_path,0,0,0,0)
print('running early full')
rf_models(analysis_path,0,1,0,0)

#run timecourse exponential pattern analysis
print('running timecourse pattern analysis')
timecourse_pattern_analysis(analysis_path)
