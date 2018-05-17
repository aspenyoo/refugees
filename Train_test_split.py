def train_test_split(path):
    #path: path where files are

    # # Train_test_split
    # We want to withhold a subset of the data to estimate the performance of the final chosen model at the end. Therefore, we currently exclude 20% of the data from any analyses. All training, validation, and model comparison will be completed on the remaining 80% of the data.  
    #  
    # Since the full asylum version is a subset of any asylum version, we create the split on the full asylum version, save the list of train and test cases,so that we can use this list to split the data for subsequent analyses.
    # this code only needs to be run one time--once we've done it, we can load the train and test cases from the file where they are stored.


    import pandas as pd
    import numpy as np
    from sklearn.cross_validation import train_test_split
    pd.set_option('precision', 5)


    # LOAD IN CLEANED DATASET
    master_app = pd.read_csv(path+'/finalmerge_any_latev2.csv', low_memory=False)


    obs_train, obs_test = train_test_split(master_app,  test_size=0.2, random_state=0)




    #check that all feature values that are present in the test set are also present in the training set.
    #categorical = obs_train.select_dtypes(include=['object'])
    #for i in categorical.columns.values.tolist():
    #    if obs_train[i].nunique() != obs_test[i].nunique():
    #        print("Feature: " + i)
    #        print("train data: "+ str(obs_train[i].nunique())+" categories.")
    #        print("test data: "+str(obs_test[i].nunique())+" categories.")




    #save a list of train and test idncases.
    train_cases = obs_train.idncase
    test_cases = obs_test.idncase
    train_cases.to_csv(path+'/train_cases_last_hearingv2.csv',index=False)
    test_cases.to_csv(path+'/test_cases_last_hearingv2.csv',index=False)

