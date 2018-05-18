def run_rf(baseline,flag_baseline,flag_early,flag_spatialonly,flag_temponly):
    
    from Cleaning_asylum import clean_master_app
    from Cleaning_detailed_schedule import clean_detailed_schedule
    from Cleaning_schedule import clean_schedule
    from time_features import make_timefeatures
    from Train_test_split import train_test_split
    from Log_Reg_Models import log_reg_models
    from Full_Model_H2O import rf_models
    import pandas as pd
    import numpy as np
    import random

    pd.set_option('precision', 5)

    #SEED THE RANDOM NUMBER GENERATOR
    random.seed(44)

    #set data paths
    analysis_path = '/home/ay963/refugees'

    #run h2o models 
    rf_models(analysis_path,1,0,0,0)
