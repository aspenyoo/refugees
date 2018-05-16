

# #### This script cleans and merges relevant variables across datasets
# #### flag_full = 1 considers proceedings/cases associated with full asylum applications. creates a dataset unique at the idncase level  (within proceeding, it picks the asylum application and within case, it picks the proceeding associated with the asylum application, prioritizing by date --most recent first)
# #### flag_full = 0 considers a grant on any asylum case type (full, witholding, wcat) a "grant" decision on the case. 
# #### Currently, it is doing cleaning and merging only for the baseline model.

# In[12]:

# ====== PACKAGES AND FILEPATHS ======
import pandas as pd
import numpy as np
pd.set_option('precision', 5)

path = '/data/Dropbox/Data/Asylum_Courts/raw'

# flags to toggle between model designs
# 2 (asylum: full, any) x 2 (predictability: early, late)

flag_full = 0
flag_early = 0

if flag_full:
    filetag = 'full'
else:
    filetag = 'any'


# ======= Clean court_appln.csv =======
# 
# relevant variables: idnProceeding, idnCase, Appl_Code

app = pd.read_csv(path + '/court_appln.csv', low_memory=False)

# making a new variable, dec, simplifying grant decisions to DENY, GRANT, or nan
app['dec']= np.nan
app.loc[app.Appl_Dec.isin(['G','F','N','L','C']),'dec']= 1
app.loc[(app["Appl_Dec"] == 'D'),'dec'] = 0
app = app[app.dec.isin([1,0])] # only include DENY or GRANT cases


if flag_full: # full asylum cases
    # only keep applications of type ASYL. sort by date within idnproceeding
    # sorting by date--if there are multiple applications with the same decision with the same case type, take the most recent one.
    app = app[app.Appl_Code.isin(['ASYL'])]

    # sort multiple times because some need to be ascending and some descending
    app = app.sort_values(['idnProceeding','Appl_Recd_Date'],ascending=[True,False])
    #len(app)

else:
    # only keep applications of type ASYL, ASYW, WCAT. sort by Grant, then deny, then case type in order (ASYL, ASYW, WCAT)
    # then date within idnproceeding
    # sorting by date--if there are multiple applications with the same decision with the same case type, take the most recent one.
    app = app[app.Appl_Code.isin(['ASYL','ASYW', 'WCAT'])]

    # sort multiple times because some need to be ascending and some descending
    app = app.sort_values(['idnProceeding','dec','Appl_Code','Appl_Recd_Date'],ascending=[True,False,True,False])


# ====== GENERATING FEATURE: NUMBER OF APPLCATIONS PER PROCEEDING & CASE ======
# BLAH CAN THIS BE CLEANER?
app['numAppsPerProc'] = 1 # placeholder
app['numAppsPerProc'] = app['numAppsPerProc'].astype('int64')
app['numAppsPerProc'] = app.groupby(['idnCase', 'idnProceeding'])['numAppsPerProc'].transform('count')

# dropping all applications with empty decisions
app = app.dropna(subset=['Appl_Dec'])

app = app.rename(columns={"idnCase":"idncase", "idnProceeding":"idnproceeding"})



# ====== PICK THE MOST RECENT GRANT APPLICATION PER PROCEEDING ======
# make unique--take the first application for each proceeding, when sorted in order dec (grant deny),
# case type(ASYL, ASYW, WCAT), date
app2 = app.groupby('idnproceeding', as_index=False).first()

# BLAH note: idnProceedingAppln was not dropped in cleaning_full_asylum
app2 = app2.drop(columns=['idnProceedingAppln', 'Appl_Dec'])


# ====== CLEAN master.csv ======
# Relevant variables: idncase, idnproceeding, osc_date, tracid, nat

# load in data
master = pd.read_csv(path + '/master.csv', low_memory=False)

# drop empty cases and proceedings
master = master.dropna(subset= ['idncase','idnproceeding'])

# stuff on osc_date (date charges filed or NTA)
master = master.dropna(subset=['osc_date']) # dropping empty dates

master['osc_date'] = master['osc_date'].astype('str')
master = master[master['osc_date'].apply(lambda x: len(x) == 9)] # delete dates invalid formats

master['osc_date'] = pd.to_datetime(master['osc_date'], format='%d%b%Y') # change to date format

# delete NTA dates before 1984
master = master[master.osc_date.dt.year>1983]


# comp date (date proceeding completed)
master = master.dropna(subset=['comp_date']) # dropping empty dates

master['comp_date'] = master['comp_date'].astype('str')
master = master[master['comp_date'].apply(lambda x: len(x) == 9)] # delete dates invalid formats

master['comp_date'] = pd.to_datetime(master['comp_date'], format='%d%b%Y') # change to date format 

#drop comp date dates before 1985
master = master[master.comp_date.dt.year>1984]



# delete duplicates (since idnproceeding are unique, this shouldn't do anything)
master = master.drop_duplicates(subset=['idncase', 'idnproceeding'])


master['idnproceeding'] = master['idnproceeding'].astype('float64')

# ======= MISSING DATA =======
#replace nan attorney flags with 0.
master.loc[pd.isnull(master.attorney_flag),'attorney_flag']=0


# ## Merge master and court_appln
merged = pd.merge(app2, master, on=['idnproceeding','idncase'])


# drop nan tracids 
merged = merged.dropna(subset=['tracid'])


# drop all cases where judge has fewer than 100 cases--same as in gambler's fallacy paper
tracid_100 = merged.groupby('tracid').idnproceeding.count() >= 100 #bool indicating whether judge has at least 100 cases
tracid_100 = tracid_100.index.values[tracid_100]#indices of judges with at least 100 cases
merged2 = merged.loc[merged.tracid.isin(tracid_100)]



# ## Load and merge tblLookupHloc and tblLookupBaseCity

# In[34]:


# mapping hearing_loc_code
hearingloc_map = pd.read_csv(path + '/tblLookupHloc.csv', header=None)


hearingloc_map = hearingloc_map.rename(columns={1:'hearing_loc_code', 5:'hearing_city'})
merged2['hearing_loc_code'] = merged2['hearing_loc_code'].astype('str').str.strip()
hearingloc_map['hearing_loc_code'] = hearingloc_map['hearing_loc_code'].astype('str').str.strip()

hearingloc_map = hearingloc_map[['hearing_loc_code', 'hearing_city']]

merged2 = pd.merge(merged2, hearingloc_map, on=['hearing_loc_code'], how='left')

merged2['hearing_city'] = merged2['hearing_city'].fillna('UNKNOWN')
merged2['hearing_loc_code'] = merged2['hearing_loc_code'].fillna('UNKNOWN')

basecity_map = pd.read_csv(path + '/tblLookupBaseCity.csv', header=None)
basecity_map = basecity_map.rename(columns={1:'base_city_code', 5:'base_city'})
basecity_map = basecity_map[['base_city_code','base_city']]


# make same dtype
merged2['base_city_code'] = merged2['base_city_code'].astype('str').str.strip()
basecity_map['base_city_code'] = basecity_map['base_city_code'].astype('str').str.strip()

# merge and fit missing data with UNKNOWN
merged2 = pd.merge(merged2, basecity_map, on=['base_city_code'], how='left')
merged2['base_city'] = merged2['base_city'].fillna('UNKNOWN')
merged2['base_city_code'] = merged2['base_city_code'].fillna('UNKNOWN')



# ## feature cleanup on the variable containing merged information

# adding additional feature based on how many asylum proceedings have been filed for the same (idnCase) 
merged2['numProcPerCase'] = 1
merged2['numProcPerCase'] = merged2['numProcPerCase'].astype('int64')
merged2['numProcPerCase'] = merged2.groupby(['idncase'])['numProcPerCase'].transform('count')

# make unique at idncase level, sorting with the same logic as used to sort applications
if flag_full:
    merged_case = merged2.sort_values(['idncase','Appl_Recd_Date'],ascending=[True,False])
else: 
    #counting case as a grant if ANY proceeding was grant
    merged_case = merged2.sort_values(['idncase','dec','Appl_Code','Appl_Recd_Date'],ascending=[True,False,True,False])
    
merged_case = merged_case.groupby('idncase',as_index=False ).first()


# get rid of merged_cases where other_comp is not null. other_comp indicates that the proceeding ended for a reason other than 
# a judge's decision, suggesting no decision was actually made. this is less than 1% of cases once we have already filtered
# out applications where the decision is not grant or deny and matched them to proceedings.
merged_case = merged_case[pd.isnull(merged_case.other_comp)]

# get rid of cases that don't have c_asy type (about 2% of cases--higher proportion than full_asylum version...)
merged_case = merged_case[~pd.isnull(merged_case.c_asy_type)] 

# change values of c_asy_type to be more clear
merged_case.loc[merged_case.c_asy_type=='I','c_asy_type'] = 'aff'
merged_case.loc[merged_case.c_asy_type=='E','c_asy_type'] = 'def'


# drop variables that definitely won't be used as features (or won't be used to track where the data came from)

merged_case = merged_case.drop(columns=['Appl_Code','Appl_Recd_Date','dec_type','other_comp','input_date','ij_code','dec_code'])

# change ?? to unknwon for  159 cases with unknown nationalities
merged_case.loc[(merged_case.nat=='??'),'nat'] = 'UNKNOWN'

# mark na nats as unknown
merged_case.loc[pd.isnull(merged_case.nat),'nat'] = 'UNKNOWN'
# load nationality lookup table
nat_lut =  pd.read_csv(path+ '/tblLookupNationality.csv',header=None)

# mark 4 observations where the nationality code is not in the lookup table as unknown
merged_case.loc[~merged_case.nat.isin(nat_lut[1]),'nat'] = 'UNKNOWN'

# mark as unknown 2 observations with nationality code XX whic the LUT says corresponds 
# to "BE REMOVED FROM THE UNITED STATES"
merged_case.loc[(merged_case.nat=="XX"),'nat'] = 'UNKNOWN'

# examine counts for different nationalities:
nat_numbers = merged_case.groupby('nat',as_index=False).idncase.count().sort_values('idncase')

# remove nationalities with less than 10 observations
nat_10 = nat_numbers.loc[nat_numbers['idncase'] > 9, 'nat']
merged_case = merged_case[merged_case.nat.isin(nat_10)]

# remove cities with less than 10 observations (only removes 2 cases)
city_numbers = merged_case.groupby('base_city_code',as_index=False).idncase.count().sort_values('idncase')
cities_10 = city_numbers.loc[city_numbers['idncase'] > 9, 'base_city_code']
merged_case = merged_case[merged_case.base_city_code.isin(cities_10)]



# hearing loc--is this "court"? some of them are prisons/detention centers/airports. many "courts" have fewer than 10 obs
# drop thewse (less than 1% of proceedings)

court_numbers = merged_case.groupby('hearing_loc_code',as_index=False).idncase.count().sort_values('idncase')

courts_10 = court_numbers.loc[court_numbers['idncase']>9,'hearing_loc_code']
merged_case = merged_case[merged_case.hearing_loc_code.isin(courts_10)]


# ====== SAVE DATA ======
merged_case.to_csv('/data/WorkData/spatialtemporal/merged_master_app_' + filetag + '.csv',index=False)

