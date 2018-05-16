

# ====== PACKAGES AND FILEPATHS ======
import relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('precision', 5)

# file paths (BLAH MAKE MORE GENERAL)
path = '/data/Dropbox/Data/Asylum_Courts/tbl_schedule'
raw_folder = '/data/Dropbox/Data/Asylum_Courts/raw'

# ====== LOAD tbl_schedule ======
cols = ["IDNSCHEDULE","IDNPROCEEDING","IDNCASE","OSC_DATE","GENERATION","SUB_GENERATION","REC_TYPE","ALIEN_ATTY_CODE","LANG","HEARING_LOC_CODE","BASE_CITY_CODE","IJ_CODE","INTERPRETER_CODE","INPUT_DATE","INPUT_TIME","UPDATE_DATE","UPDATE_TIME","ASSIGNMENT_PATH","CONTINUE_FLAG","CAL_TYPE","ADJ_DATE","ADJ_TIME_START","ADJ_TIME_STOP","ADJ_RSN","ADJ_MEDIUM","ADJ_MSG","ADJ_ELAP_DAYS","ID_1","LNGSESSNID","SCHEDULE_TYPE","NOTICE_CODE","DATBATCHMODIFIED","STRCREATEDBY","STRMODIFIEDBY","BLNCLOCKOVERRIDE","EOIRAttorneyID"]
cols = [x.lower() for x in cols]

sched_tbl = pd.read_csv(path + '/tbl_schedule_01.csv', header=None, names=cols, encoding='ISO-8859-1', low_memory=False, dtype={'adj_date':'str', 'adj_time_start':'str', 'adj_time_stop':'str'})


sched_tbl = sched_tbl[['idnschedule','idnproceeding','idncase','osc_date','cal_type', 'lang','hearing_loc_code', 'base_city_code', 'alien_atty_code', 'adj_date','adj_time_start','adj_time_stop','adj_rsn','schedule_type', 'notice_code','eoirattorneyid']]

sched_tbl = sched_tbl.dropna(subset=['idnschedule', 'idnproceeding', 'idncase', 'adj_date'])


# ====== LANGUAGE OF HEARING =======
lang_map = pd.read_excel('../tbl_language.xlsx')

# change to string
sched_tbl['lang'] = sched_tbl['lang'].astype('str').str.strip()
lang_map['lang'] = lang_map['lang'].astype('str').str.strip()

# merge tbl_schedule with language data set
sched_tbl = pd.merge(sched_tbl, lang_map, on=['lang'], how='left')

# mark the ones that don't match as UNKNOWN
sched_tbl['lang_hearing'] = sched_tbl['lang_hearing'].fillna('UNKNOWN LANGUAGE')



# ====== HEARING SCHEDULE TYPE ======
schedtype_map = pd.read_csv(raw_folder + '/tbllookupSchedule_Type.csv', header=None)

schedtype_map = schedtype_map.rename(columns={1:'schedule_type', 2:'sched_type'})
schedtype_map = schedtype_map[['schedule_type', 'sched_type']]

sched_tbl['schedule_type'] = sched_tbl['schedule_type'].astype('str').str.strip()
schedtype_map['schedule_type'] = schedtype_map['schedule_type'].astype('str').str.strip()

sched_tbl = pd.merge(sched_tbl, schedtype_map, on=['schedule_type'], how='left')

sched_tbl = sched_tbl.drop(columns=['schedule_type', 'base_city_code'])


# MERGING notice_code (BLAH WHAT IS THIS?) ONTO sched_tbl
notice_map = pd.read_csv(raw_folder + '/eoir/tblLookupNOTICE.csv', header=None)

# take only relevant variables
notice_map = notice_map.rename(columns={1:'notice_code', 2:'notice_desc'})
notice_map = notice_map[['notice_code', 'notice_desc']]

# change both to strings
sched_tbl['notice_code'] = sched_tbl['notice_code'].astype('str').str.strip()
notice_map['notice_code'] = notice_map['notice_code'].astype('str').str.strip()

# merge notice_map with sched_tbl
sched_tbl = pd.merge(sched_tbl, notice_map, on=['notice_code'], how='left')

# fill NAs as UNKNOWN
sched_tbl['notice_desc'] = sched_tbl['notice_desc'].fillna('UNKNOWN')

# drop irrelevant variables
sched_tbl = sched_tbl.drop(columns=['notice_code', 'eoirattorneyid', 'cal_type'], axis=1)


# ====== CREATE HEARING DURATION VARAIBLE: durationHearing =======

sched_tbl['adj_date'] = pd.to_datetime(sched_tbl['adj_date'])

sched_tbl['adj_time_start'] = sched_tbl['adj_time_start'].str.zfill(4)
sched_tbl['adj_time_stop'] = sched_tbl['adj_time_stop'].str.zfill(4)

sched_tbl['adj_time_start'] = pd.to_datetime(sched_tbl['adj_time_start'], format='%H%M', errors='coerce')
sched_tbl['adj_time_stop'] = pd.to_datetime(sched_tbl['adj_time_stop'], format='%H%M', errors='coerce')

# extract adj_date year-month-day portion
d = sched_tbl['adj_date'].dt.strftime('%Y-%m-%d')

# BLAH WHY DO WE DO THIS?
sched_tbl.loc[sched_tbl.adj_time_start.dt.hour == 0, 'adj_time_start'] = pd.NaT
sched_tbl.loc[sched_tbl.adj_time_stop.dt.hour == 0, 'adj_time_stop'] = pd.NaT

# BLAH WHY DO WE DO THIS?
sched_tbl.loc[sched_tbl.adj_time_start.dt.hour < 6, 'adj_time_start'] += pd.to_timedelta(12, unit='h')
sched_tbl.loc[sched_tbl.adj_time_stop.dt.hour < 6, 'adj_time_stop'] += pd.to_timedelta(5, unit='h')

# start and stop time in hours, minutes, seconds
t_start = sched_tbl['adj_time_start'].dt.strftime('%H:%M:%S')
t_stop = sched_tbl['adj_time_stop'].dt.strftime('%H:%M:%S')

# make variable that combines date and time information about hearing start and stop times
sched_tbl['adj_time_start2'] = pd.to_datetime(d + " " + t_start, errors='coerce')
sched_tbl['adj_time_stop2'] = pd.to_datetime(d + " " + t_stop, errors='coerce')

# define hearing duration as the time between the start and top time in minutes
sched_tbl['durationHearing'] = sched_tbl['adj_time_stop2'] - sched_tbl['adj_time_start2']
sched_tbl['durationHearing'] = sched_tbl['durationHearing'].astype('timedelta64[m]')

# drop negative hearing durations
sched_tbl = sched_tbl[sched_tbl.durationHearing >= 0]

# drop variables no longer relevant
sched_tbl = sched_tbl.drop(columns=['adj_time_start', 'adj_time_stop'], axis=1)



# ====== MERGE ADJOURNMENT REASONS: adj_rsn ======
adjrsn_map = pd.read_excel(raw_folder+'/dbo_tblAdjournmentcodes.xlsx')

adjrsn_map = adjrsn_map.rename(columns={'strcode':'adj_rsn', 'strDesciption':'adj_rsn_desc'})
adjrsn_map = adjrsn_map[['adj_rsn', 'adj_rsn_desc']]

sched_tbl['adj_rsn'] = sched_tbl['adj_rsn'].astype('str').str.strip()
adjrsn_map['adj_rsn'] = adjrsn_map['adj_rsn'].astype('str').str.strip()

sched_tbl = pd.merge(sched_tbl, adjrsn_map, on=['adj_rsn'], how='left')

# mark the ones that don't match as UNKNOWN
sched_tbl['adj_rsn_desc'] = sched_tbl['adj_rsn_desc'].fillna('UNKNOWN')



# ====== OUTPUT CLEANED FILE ======
sched_tbl.to_csv('../detailed_schedule.csv', index=False)

