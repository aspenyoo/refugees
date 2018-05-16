
# ## Merging schedule.csv, merged_master_app.csv, and tbl_schedule.csv
# Merging all four files together. Please note that master_app already took care of master and application files.

# ====== PACKAGES, FILEPATHS, FLAGS ======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('precision', 5)

path = '/data/Dropbox/Data/Asylum_Courts/raw'
rubypath = '/home/yihsuanfu/'

# flags to toggle between modes
# 2 (asylum: full, any) x 2 (predictability: early, late)

flag_full = 0 # is full asylum? if not, then any asylum
flag_early = 1 # is early predictability? if not, then late predictability

if flag_early:
    tag_pred = 'early'
else:
    tag_pred = 'late'
    
if flag_full:
    tag_asyl = 'full'
else:
    tag_asyl = 'any'


# ====== master_app ======
master_app = pd.read_csv('/data/WorkData/spatialtemporal/merged_master_app_' + tag_asyl + '.csv'
                         
# ====== MERGE schedule.csv AND tbllookupSchedule_Type.csv ======
# load schedule.csv
sched = pd.read_csv(path + '/schedule.csv', low_memory=False)
                         
# delete dates invalid formats
sched['adj_date'] = sched['adj_date'].astype('str')
sched = sched[sched['adj_date'].apply(lambda x: len(x) == 9)] 
sched['adj_date'] = pd.to_datetime(sched['adj_date'], format='%d%b%Y')

# remove all rows that have dates beyond 2013
sched = sched[sched.adj_date.dt.year <= 2013]
                         
# load tbllookupSchedule_Type.csv to get schedule type
schedtype_map = pd.read_csv(path + '/tbllookupSchedule_Type.csv', header=None)
schedtype_map = schedtype_map.rename(columns={1:'schedule_type', 2:'sched_type'})
schedtype_map = schedtype_map[['schedule_type', 'sched_type']]

# make same dtype
sched['schedule_type'] = sched['schedule_type'].astype('str').str.strip()
schedtype_map['schedule_type'] = schedtype_map['schedule_type'].astype('str').str.strip()

# merge file, then drop variables used to merge
sched = pd.merge(sched, schedtype_map, on=['schedule_type'], how='left')
sched = sched.drop(columns=['schedule_type', 'idnschedule'])

# set NAs to UNKNOWN
sched['adj_medium'] = sched['adj_medium'].fillna('UNKNOWN')


# ====== LOADING AND CLEANING detailed_schedule.csv =======
tbl_sched = pd.read_csv(rubypath + 'detailed_schedule.csv', low_memory=False)
tbl_sched['adj_date'] = pd.to_datetime(tbl_sched['adj_date'])

tbl_sched = tbl_sched[tbl_sched.adj_date.dt.year <= 2013] # remove all rows beyond 2013
tbl_sched = tbl_sched.drop(columns=['idnschedule', 'hearing_loc_code', 'adj_rsn', 'sched_type', 'alien_atty_code', 'osc_date'])

                         
# ====== MERGE FILES ======
# merging master_app.csv with schedule.csv
merged_sched_master = pd.merge(master_app, sched, on=['idncase', 'idnproceeding'])

# merging with detailed_schedule.csv
merged_sched_master = pd.merge(merged_sched_master, tbl_sched, on=['idncase', 'idnproceeding', 'adj_date'], how='left')

                         
# ====== GENERATING FEATURES THAT DEPEND ON NUMBER OF HEARINGS =======
merged_sched_master = merged_sched_master.sort_values(['idnproceeding','adj_date']).reset_index()

if not flag_early:
    # NUMBER OF HEARINGS PER PROCEEDING 
    merged_sched_master['numHearingsPerProc'] = 1
    merged_sched_master['numHearingsPerProc'] = merged_sched_master.groupby(['idncase', 'idnproceeding'])['numHearingsPerProc'].transform('count')

    # NUMBER OF DAYS BETWEEN FIRST AND LAST HEARING
    merged_sched_master['durationFirstLastHearing']= merged_sched_master.groupby(['idncase', 'idnproceeding'])['adj_date'].transform(lambda x: x.iloc[-1] - x.iloc[0]).dt.days
    # this shows that many proceedings spanned an unreasonable amount of time (in days)
    merged_sched_master.sort_values(['durationFirstLastHearing', 'idnproceeding', 'adj_date'], ascending=False).head(20)

    # AVERAGE HEARING DURATION 
    merged_sched_master['averageHearingDur']= merged_sched_master.groupby(['idncase', 'idnproceeding'])['durationHearing'].transform('mean')

merged_sched_master = merged_sched_master.drop(columns=['index', 'lang'])

                         
# ====== SELECT RELEVANT HEARING ======
# The first (for early) or last (for late) hearing for each idncase

if flag_early: # early predictability
    merged_sched_master = merged_sched_master.groupby(['idncase', 'idnproceeding'], as_index=False).first()
else: # late predictability
    merged_sched_master = merged_sched_master.groupby(['idncase', 'idnproceeding'], as_index=False).last()


# ====== GENERATING FEATURES THAT ARE SPECIFIC TO ONE HEARING =======

# osc_date_delta: number of days from 01/01/1984 to osc date in days
merged_sched_master['osc_date'] = pd.to_datetime(merged_sched_master['osc_date'],infer_datetime_format = True)
startdate = np.datetime64('1984-01-01') # earliest osc_date in training set
merged_sched_master['osc_date_delta'] = merged_sched_master['osc_date'].apply(lambda x: (x - startdate).days)

# case duration: number of days between osc_date and comp_date (only for late predictability)
if not flag_early: 
    merged_sched_master['comp_date'] = pd.to_datetime(merged_sched_master['comp_date'])
    #merged_sched_master['osc_date'] = pd.to_datetime(merged_sched_master['osc_date'])
    merged_sched_master['caseDuration'] = merged_sched_master.apply(lambda x: x['comp_date'] - x['osc_date'], axis=1).dt.days

# hearing day of the week, year, month
merged_sched_master['hearingDayOfWeek'] = merged_sched_master['adj_date'].dt.weekday_name
merged_sched_master['hearingYear'] = merged_sched_master['adj_date'].dt.year
merged_sched_master['hearingMonth'] = merged_sched_master['adj_date'].dt.month

# political affiliation of president during hearing
pres = {1984:'REP', 1985:'REP', 1986:'REP', 1987:'REP', 1988:'REP', 1989:'REP', 1990:'REP', 1991:'REP', 1992:'REP', 1993:'DEM', 1994:'DEM', 1995:'DEM', 1996:'DEM', 1997:'DEM', 1998:'DEM', 1999:'DEM', 2000:'DEM', 2001:'REP', 2002:'REP', 2003:'REP', 2004:'REP', 2005:'REP', 2006:'REP', 2007:'REP', 2008:'REP', 2009:'DEM', 2010:'DEM', 2011:'DEM', 2012:'DEM', 2013:'DEM', 2014:'DEM'}
merged_sched_master['pres_aff'] = merged_sched_master['adj_date'].dt.year.map(pres)
                         
                         
# ====== MISSING DATA ======
# fill missing data with unknown
merged_sched_master['lang_hearing'] = merged_sched_master['lang_hearing'].fillna('UNKNOWN')
                         
# imputing missing data with mean
merged_sched_master['numAppsPerProc'].fillna((merged_sched_master['numAppsPerProc'].mean()), inplace=True)
merged_sched_master['numProcPerCase'].fillna((merged_sched_master['numProcPerCase'].mean()), inplace=True)
merged_sched_master['durationHearing'].fillna((merged_sched_master['durationHearing'].mean()), inplace=True)
if not flag_early: # (variables used in only late predictability model)
    merged_sched_master['durationFirstLastHearing'].fillna((merged_sched_master['durationFirstLastHearing'].mean()), inplace=True)
    merged_sched_master['averageHearingDur'].fillna((merged_sched_master['averageHearingDur'].mean()), inplace=True)
    merged_sched_master['numHearingsPerProc'].fillna((merged_sched_master['numHearingsPerProc'].mean()), inplace=True)


# ====== SAVE FINAL MERGE FILE ======
merged_sched_master.to_csv('/data/WorkData/spatialtemporal/finalmerge_'+ tag_asyl + '_' + tag_pred + 'v2.csv', index=False)

