
# # Time features
# ## This script creates  grant rate features, eg. grant rate in the past year
#for a particular nationality, or for a particular judge.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('precision', 5)


#FUNCTIONS THAT GENERATE FEATURES
def recent_g_hist_yrs_el(df,feat,yrs,date_feat):
    #creates features indicating recent grant history for a given feature of interest (eg nationality, judge ID),
    #for a specified number of years.
        #Args:
            #df, dataframe with observations
            #feat, feature by which we use to group decisions, like judge or nat
            # yrs, the time window (currently in years) (note, have only tested 1 year periods because that was the
                # period we decided on for project)
            #date_feat, feature name that has date (comp_date or osc_date, depending on the timepoint we're predicting from.
                # if generating features for "early" model, use osc_date. for late, use comp_date.
        #returns:
            #df, dataframe with new features added (grant rate in spec period, num dec in spec period,
            #dummy feature indicating if there are no observations for the specified time period)

    #save df as df_orig then just take relevant variables from df to conserve memory
    df_orig = df
    df = df[['idnproceeding','dec','osc_date','comp_date',feat]]


    #determine whether we are calculating grant rate at early or late timepoint within case.
    if date_feat=='osc_date':
        earlylate='early'
    elif date_feat == 'comp_date':
        earlylate = 'late'

    count_feat_name = feat+'_num_last' + str(yrs) +'yr_'+ earlylate#name of new decision count feature
    gr_feat_name = feat + '_gr_last' + str(yrs) + 'yr_' + earlylate#name of new grant rate feature
    dummy_name = feat+'_dmy_last' + str(yrs) +'yr_' + earlylate #name of dummy feature

    #get unique values of feature of interest
    unique_feat = df[feat].unique()

    #cycle through unique feature values (e.g., cycle through all the nationalities)
    for f in unique_feat:
        dates = df.loc[df[feat]==f,date_feat].reset_index(drop=True) #dates associated with each case for a given feature
        dates_1back = dates - pd.DateOffset(years=yrs) #calculate dates minus given num years
        df_feat = df.loc[df[feat]==f]#df just with this specific feature (e.g., df with just cases for one nationality)
        num_lastyear = np.zeros(len(dates))#for the specific feature, will store number of decisions in the year before this decision
        gr_lastyear = np.zeros(len(dates))#for the specific feature, will store grant rate in the year before this decision

        #cycle through all the cases associated with this specific feature, and  calculate grant rate
        #in the last timeperiod, and num decisions
        for i in range(len(dates)):
            #calc grant rate after specified num years ago but before today
            #note:
            #date_feat (osc_date or comp_date) determines, for a given decision, at what point in time we are
            #checking the grant rate. but comp_date, not osc_date should be used to determine which
            #decisions are included in history.
            gr_lastyear[i] = np.mean(df_feat.loc[(df_feat['comp_date'] > dates_1back[i])
                                             & (df_feat['comp_date'] < dates[i]),'dec'])
            num_lastyear[i] = len(df_feat.loc[(df_feat['comp_date'] > dates_1back[i])
                                             & (df_feat['comp_date'] < dates[i]),'dec']) # num decisions in the last timeperiod


        #store this info in main df
        df.loc[df[feat]==f,count_feat_name] = num_lastyear
        df.loc[df[feat]==f,gr_feat_name] = gr_lastyear


    #now match these new features back to the original data frame
    df = pd.merge(df_orig,df[['idnproceeding',gr_feat_name,count_feat_name]],how='left',on=['idnproceeding'])

    #make dummy variable indicating if there are no observations for the specified time period
    #and replace nan grant rates with mean
    df[dummy_name] = np.zeros(len(df))
    df.loc[np.isnan(df[gr_feat_name]),dummy_name] = 1 #flag nans with dummy variable
    df.loc[np.isnan(df[gr_feat_name]),gr_feat_name] = np.mean(df[gr_feat_name])#replace nans with mean

    return(df)


def recent_g_hist_n_d(df,feat,n_d):
    #creates features indicating recent grant history for a given feature of interest (eg nationality, judge ID),
    #for a specified number of recent decisions
        #Args:
            #df, dataframe with observations
            #feat, feature by which we use to group decisions, like judge or nat
            # n_d, the time window (measured in number of decisions)
        #returns:
            #df, dataframe with new features added (grant rate in spec period, dummy for when there
                #aren't n_d decisions in history)


    #new feature names
    gr_feat_name = feat + '_gr_last' + str(n_d) + 'd'#name of new grant rate feature
    dummy_name = feat+'_dmy_last' + str(n_d) + 'd'


    #only keep relevant variables
    df_orig = df
    df = df[['idnproceeding','dec','timeLastHearing','comp_date',feat]]

    #drop observations where time data is missing (nan)
    #because cannot determine order (added back in later when merge with df_orig).
    df = df.dropna(subset=['timeLastHearing'])

    #group by feature, comp date, time last hearing for computation of recent grant rate (because not going to count
    #decisions that ocurred with the same judge, comp date and time twice)
    #take the mean of the decisions if there are multiples.
    df = df.groupby([feat,'comp_date','timeLastHearing'],as_index=False).dec.mean()


    nanlist = np.empty(len(df))
    nanlist[:] = np.nan
    df[gr_feat_name]=nanlist #set this column to nans, and we'll fill it in with values where there isn't missing data.

    #get unique values of feature of interest
    unique_feat = df[feat].unique()

    #cycle through unique feature values (eg, each nationality)
    for f in unique_feat:

        df_feat = df.loc[df[feat]==f].reset_index(drop=True)#df for each specific feature

        nanlist = np.empty(len(df_feat))
        nanlist[:] = np.nan
        gr_rec = nanlist
        #cycle through decisions and  calculate grant rate in the last timeperiod
        for i in range(len(df_feat)):
            if i>=n_d: #if there are at least n_d decisions in the history
                #take the mean of last 10 (or n_d) decisions:
                last10dec =df_feat.loc[(i-n_d):(i-1),'dec']
                gr_rec[i] = np.mean(last10dec)

        #store this info in df
        df.loc[df[feat]==f,gr_feat_name] = gr_rec

    #now match these new features back to the original data frame, using feature (judge), comp date, time
    df = pd.merge(df_orig,df[[feat,'comp_date','timeLastHearing',gr_feat_name]],
                  how='left',on=[feat,'comp_date','timeLastHearing'])

    #make dummy variable and replace nans with mean
    df[dummy_name] = np.zeros(len(df))
    df.loc[np.isnan(df[gr_feat_name]),dummy_name] = 1 #flag nans with dummy variable
    df.loc[np.isnan(df[gr_feat_name]),gr_feat_name] = np.mean(df[gr_feat_name])#replace nans with mean

    return(df)

def g_hist_by_n_d(df,feat,n_d,n_p):
    #creates features indicating  grant history for a given feature of interest (eg nationality, judge ID),
    #for a specified number of recent decisions, for a given number of windows back
    #eg n_d is 10, n_p is 3, would produce features assessing grant rate for the last 10 decisions, the 10 before that,
    #and the 10 before that.
        #Args:
            #df, dataframe with observations
            #feat, feature by which we use to group decisions, like judge or nat
            # n_d, the time window (measured in number of decisions)
            # n_p, the number of periods back to look
        #returns:
            #df, dataframe with new features added (grant rate in spec period, dummy for nans)


    df_orig = df

    #only keep relevant variables
    df = df[['dec','timeLastHearing','comp_date',feat]]


    #drop time nans because cannot determine order (added back in later when merge with df_orig).
    df = df.dropna(subset=['timeLastHearing'])



    #group by feature, comp date, time last hearing for computation of recent grant rate (because not going to count
    #decisions that ocurred with the same judge, comp date and time twice)
    #take the mean of the decisions if there are multiples.
    df = df.groupby([feat,'comp_date','timeLastHearing'],as_index=False).dec.mean()



    #make grant rate feature names and initialize with nans
    gr_ft_names = []
    dummy_names = []
    nanlist = np.empty(len(df))
    nanlist[:] = np.nan
    for i in range(n_p):
        gr_ft_names.append(feat + '_gr_' +  str(i*n_d+1) + '_' + str((i+1)*n_d) + 'd')#name of new grant rate feature
        df[gr_ft_names[i]]=nanlist #set this column to nans, and we'll fill it in with values where there isn't missing data.
        dummy_names.append(feat + '_dmy_' + str(i*n_d+1) + '_' + str((i+1)*n_d) + 'd')


    #get unique values of feature of interest
    unique_feat = df[feat].unique()


    #cycle through unique feature values
    for f in unique_feat:

        df_feat = df.loc[df[feat]==f].reset_index(drop=True)
        nanlist = np.empty((len(df_feat),n_p))
        nanlist[:] = np.nan
        gr_rec = nanlist

        #cycle through decisions and  calculate grant rate in the last timeperiod
        for i in range(len(df_feat)):

            for p in range(n_p):#for each period
                dec_start = (p+1)*n_d# number of decisions ago for stop of period
                dec_stop = p*n_d+1# number of decisions ago for start of period
                if i>=dec_start: #if there are at least dec_start decisions in the history
                    dec_period =df_feat.loc[(i-dec_start):(i-dec_stop),'dec'] #take the mean of decisions in this period
                    gr_rec[i,p] = np.mean(dec_period)

        #store this info in df
        df.loc[df[feat]==f,gr_ft_names] = gr_rec
    df = df.drop('dec',axis=1)

    #now match these new features back to the original data frame, using feature (judge), comp date, time
    df = pd.merge(df_orig,df,how='left',on=[feat,'comp_date','timeLastHearing'])


    for i in range(n_p):
        df[dummy_names[i]] = np.zeros(len(df))
        df.loc[np.isnan(df[gr_ft_names[i]]),dummy_names[i]] = 1 #flag nans with dummy variable
        df.loc[np.isnan(df[gr_ft_names[i]]),gr_ft_names[i]] = np.mean(df[gr_ft_names[i]])#replace nans with mean
    return(df)


def g_hist_by_period(df,feat,yrs,min_d):
    #creates features indicating grant rate through time. for a given observation, makes a feature with grant rate
    #for a given category in the last year, and the year before that, and the year before that (number of years
    #per period can be specified in arg).HAVENT TESTED FOR ANY PERIOD OTHER THAN 1 YR though
        #Args:
            #df, dataframe with observations
            #feat, feature by which we use to group decisions, like judge or nat
            # yrs, the time window in years
            #min_d, minimum number of decisions for grant rate to not be considered "missing"
        #returns:
            #df, dataframe with new features added (grant rate in spec period, num dec in spec period, dummy for nans)


     #group by feature, then by date, and record the number deny and number grant for that nat/date combo.
    #call this dec_counts, with new variables "g" and "d" wiht the counts.
    # all decisions on a given day will be assigned the same rate for the last year (or whatever the chosen period is).
    #thus, we can reduce computation by only computing the rate once or each grouping.
    dec_counts = df.groupby([feat,'comp_date'],as_index=False).dec.agg({'g': lambda x: len(x[x ==1]),
                                                                             'd': lambda x: len(x[x ==0])})
    #get unique values of feature of interest
    unique_feat = dec_counts[feat].unique()

    #determine total number timeperiods to be examined, by finding the category with the biggest time range:
    max_range = df.groupby([feat]).comp_date.apply(lambda x: max(x.dt.year)-min(x.dt.year))
    num_periods = int(round(max(max_range)/ yrs)-1)

    #make grant rate feature names
    gr_ft_names = []
    count_ft_names = []
    dummy_names = []
    for i in range(num_periods):
        gr_ft_names.append(feat + '_gr_' + str((num_periods-i)*yrs) + 'yr')#name of new grant rate feature
        dec_counts[gr_ft_names[i]] = np.zeros(len(dec_counts))
        count_ft_names.append(feat + '_num_d_' + str((num_periods-i)*yrs) + 'yr')
        dec_counts[count_ft_names[i]] = np.zeros(len(dec_counts))
        dummy_names.append(feat + '_dmy_' + str((num_periods-i)*yrs) + 'yr')


    #cycle through unique feature values
    for f in unique_feat:
        print(f)
        dates = dec_counts.loc[dec_counts[feat]==f,'comp_date'].reset_index(drop=True) #dates of observations for a given category
        num_d_period = np.zeros((len(dates),num_periods))
        gr_period = np.zeros((len(dates),num_periods))
        for p in range(num_periods):#cycle through all the periods.
            n_p_back = num_periods-p#calculate the number of periods back corresponding to this period.
            #(ie, if this is period 1 of 30 periods, then it's 29 periods back. if it's period 2, it's 28 periods back)

            #calculate start and stop date of each period for each observation.
            dates_nback_start = dates - pd.DateOffset(years=(yrs*n_p_back)) #calculate start date of present - n years
            dates_nback_stop = dates - pd.DateOffset(years=(yrs*(n_p_back-1)))
            dec_counts_feat = dec_counts.loc[dec_counts[feat]==f]


            #cycle through dates and  calculate grant rate in the last timeperiod, and num decisions
            for i in range(len(dates)):
                #count num grants between start and stop date
                g_period = np.sum(dec_counts_feat.loc[(dec_counts_feat['comp_date'] > dates_nback_start[i])
                                                 & (dec_counts_feat['comp_date'] < dates_nback_stop[i]),'g'])
                #count num denies between start and stop date
                d_period = np.sum(dec_counts_feat.loc[(dec_counts_feat['comp_date'] > dates_nback_start[i])
                                                 & (dec_counts_feat['comp_date'] < dates_nback_stop[i]),'d'])
                num_d_period[i,p] = g_period + d_period # num decisions in the  timeperiod
                gr_period[i,p] = g_period / num_d_period[i,p] #grant rate in the  timeperiod
        #store this info in  dec_counts
        dec_counts.loc[dec_counts[feat]==f,count_ft_names] = num_d_period
        dec_counts.loc[dec_counts[feat]==f,gr_ft_names] = gr_period



    #now match these new features back to the original data frame
    dec_counts = dec_counts.drop(['d','g'],axis=1)
    df = pd.merge(df,dec_counts,how='left',on=[feat,'comp_date'])

    for i in range(num_periods):
        df.loc[df[count_ft_names[i]]<min_d,gr_ft_names[i]] = np.nan #flag entries with less than min_d decisions  as missing
        df[dummy_names[i]] = np.zeros(len(df))
        df.loc[np.isnan(df[gr_ft_names[i]]),dummy_names[i]] = 1 #flag nans with dummy variable
        df.loc[np.isnan(df[gr_ft_names[i]]),gr_ft_names[i]] = np.mean(df[gr_ft_names[i]])#replace nans with mean
    return(df)


def main():
    #read in data
    merged_data = pd.read_csv('/data/WorkData/spatialtemporal/merged_last_hearing.csv')

    #convert dates to datetime format
    merged_data['osc_date'] = pd.to_datetime(merged_data['osc_date'],infer_datetime_format = True)
    merged_data['comp_date'] = pd.to_datetime(merged_data['comp_date'],infer_datetime_format = True)
    merged_data['adj_time_stop2'] = pd.to_datetime(merged_data['adj_time_stop2'],infer_datetime_format = True)
    #make a time of day feature that is just a time, not a date:
    merged_data['timeLastHearing'] = merged_data['adj_time_stop2'].dt.time



    #make features indicating grant rate in the last year based on nationality, judge, and base city,
    #at both early and late timepoints
    merged_data = recent_g_hist_yrs_el(merged_data,'nat',1,'osc_date')
    merged_data = recent_g_hist_yrs_el(merged_data,'nat',1,'comp_date')
    merged_data = recent_g_hist_yrs_el(merged_data,'tracid',1,'osc_date')
    merged_data = recent_g_hist_yrs_el(merged_data,'tracid',1,'comp_date')
    merged_data = recent_g_hist_yrs_el(merged_data,'base_city_code',1,'osc_date')
    merged_data = recent_g_hist_yrs_el(merged_data,'base_city_code',1,'comp_date')
    merged_data.to_csv('/home/emilyboeke/temp_timecourses.csv',index=False)


    #make feature with the grant rate for the last 10 decisisons for a given judge
    merged_data = recent_g_hist_n_d(merged_data,'tracid',10)


    #split up "early" and "late" timecourses to go in separate files
    tc_early = merged_data[['idnproceeding','idncase','nat_num_last1yr_early','nat_gr_last1yr_early',
                             'nat_dmy_last1yr_early','tracid_num_last1yr_early','tracid_gr_last1yr_early',
                             'tracid_dmy_last1yr_early','base_city_code_num_last1yr_early','base_city_code_gr_last1yr_early',
                             'base_city_code_dmy_last1yr_early']]

    tc_late = merged_data[['idnproceeding','idncase','nat_num_last1yr_late','nat_gr_last1yr_late',
                             'nat_dmy_last1yr_late','tracid_num_last1yr_late','tracid_gr_last1yr_late',
                             'tracid_dmy_last1yr_late','base_city_code_num_last1yr_late','base_city_code_gr_last1yr_late',
                             'base_city_code_dmy_last1yr_late','tracid_num_last1yr_late','tracid_gr_last10d','tracid_dmy_last10d']]


    tc_early.to_csv('/data/WorkData/spatialtemporal/gr_lastyear_early.csv',index=False)
    tc_late.to_csv('/data/WorkData/spatialtemporal/gr_lastyear_late.csv',index=False)


    #the features made above only calculate grant rate for the most recent period.
    #now, make features with grant rate for multiple periods back

    #make features with grant rate for each judge, for the last 10 decisions back, 10 before that, 10 before that, etc
    #for 15 periods
    merged_data  = g_hist_by_n_d(merged_data,'tracid',10,15)

    #make features indicating grant rate for a given nationality for every year period in the past
    merged_data = g_hist_by_period(merged_data,'nat',1,30)

    #save these features in a separate file
    tc_late_extra = pd.concat((merged_data[['idnproceeding','idncase']],merged_data.loc[:,'tracid_gr_1_10d':'nat_dmy_1yr']),axis=1)
    tc_late_extra.to_csv('/data/WorkData/spatialtemporal/extra_timefeatures_late.csv',index=False)

if __name__ == "__main__":
    main()
