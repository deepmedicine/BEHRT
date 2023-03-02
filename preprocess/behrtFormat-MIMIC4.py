from datetime import date
import pandas as pd
import numpy as np

def age(admission_date,anchor_date,anchor_age):
    age = admission_date.year - anchor_date.year - ((admission_date.month, admission_date.day) < (anchor_date.month, anchor_date.day)) + anchor_age
    return age

df_adm = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-2.1/hosp/admissions.csv')
df_adm.admittime = pd.to_datetime(df_adm.admittime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm.dischtime = pd.to_datetime(df_adm.dischtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm.deathtime = pd.to_datetime(df_adm.deathtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')

df_adm = df_adm.sort_values(['subject_id', 'admittime'])

df_pat = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-2.1/hosp/patients.csv')

df_adm = df_adm.merge(df_pat, how='inner', on='subject_id')

df_main = df_adm[['subject_id','hadm_id',
'admittime','dischtime','deathtime','anchor_age']]

anchor_time = df_main.groupby('subject_id')['admittime'].min().reset_index()

df_main = df_main.merge(anchor_time, how='left', on='subject_id')


### admittime_x = date on admission
### admittime_y = date for anchor_age
df_main['age_on_admittance'] = df_main.apply(lambda x: age(x['admittime_x'],x['admittime_y'],x['anchor_age']), axis=1)



print(df_main)

