from datetime import date
import pandas as pd
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql import Window


#Create PySpark SparkSession
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("SparkByExamples.com") \
    .getOrCreate()



def calculate_age_on_current_admission(admission_date,anchor_time,anchor_age):
    age = admission_date.year - anchor_time.year - ((admission_date.month, admission_date.day) < (anchor_time.month, anchor_time.day)) + anchor_age
    return age  

# MIMIC IV
df_adm = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-2.1/hosp/admissions.csv')
df_pat = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-2.1/hosp/patients.csv')

# MIMIC IV ED
df_edstays = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-ed-2.0/2.0/ed/edstays.csv')
df_eddiagnosis = pd.read_csv('/home/josaphat/Desktop/research/mimic-iv-ed-2.0/2.0/ed/diagnosis.csv')

# taking relevant columns from MIMIC-IV-ED
df_edstays = df_edstays[['subject_id','hadm_id','stay_id','intime','outtime','arrival_transport','disposition']]

# transform column values to processable datetime
df_adm.admittime = pd.to_datetime(df_adm.admittime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm.dischtime = pd.to_datetime(df_adm.dischtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')    
df_adm.deathtime = pd.to_datetime(df_adm.deathtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_edstays.intime = pd.to_datetime(df_edstays.intime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_edstays.outtime = pd.to_datetime(df_edstays.outtime, format = '%Y-%m-%d %H:%M:%S', errors='coerce')
df_adm = df_adm.sort_values(['subject_id', 'admittime'])

# merge admission info with patient demographics info
df_adm = df_adm.merge(df_pat, how='inner', on='subject_id')

# taking relevant columns, save it as Main Dataframe
df_main = df_adm[['subject_id','hadm_id',
'admittime','dischtime','deathtime','anchor_age']]

# find the first time patient is admitted to the hospital, save it in anchor_time
anchor_time = df_main.groupby('subject_id')['admittime'].min().reset_index()
anchor_time = anchor_time.rename(columns={'admittime':'anchor_time'})
df_main = df_main.merge(anchor_time, how='left', on='subject_id')

# merge patient info from MIMIC-IV with MIMIC-IV-ED edstays on same ID
df_main = df_main.merge(df_edstays, how='inner', on = ['subject_id','hadm_id'])

# calculate patient age during admission
### admittime = date on admission
### anchor_time = date when anchor_age is given
df_main['age_on_admittance'] = df_main.apply(lambda x: calculate_age_on_current_admission(x['admittime'],x['anchor_time'],x['anchor_age']), axis=1)

# merge ED admission with ED diagnosis on same patient_id and admission_id
df_main = df_main.merge(df_eddiagnosis, how='inner', on=['subject_id','stay_id'])
df_main2 = df_main[['subject_id','intime','age_on_admittance','icd_code']]

# transform dataframe into spark due to unavailable method on normal pandas
sparkDF=spark.createDataFrame(df_main2)
sparkDF = sparkDF.groupBy(['subject_id','intime']).agg(F.collect_list('age_on_admittance').alias('age_on_admittance'),F.collect_list('icd_code').alias('icd_code'))

### diagnoses = diagnoses.groupby(['patid', 'eventdate']).agg(F.collect_list(config['col_name']).alias(config['col_name']), F.collect_list('age').alias('age'), F.first('yob').alias('yob'))

print(sparkDF.head())

config= {
    'diagnoses': '',  # data path for diagnoses/medication
    'demographic': '',  # data path for demographic information
    'output': '',  # path to save formated file
    'col_name': ''  # column name for ICD/Med code
}

df_main = sparkDF.toPandas()

def array_add_element(array, val):
    return array + [val]

print(df_main.head())

df_main['icd_code'] = df_main.apply(lambda row: array_add_element(row['icd_code'], 'SEP'),axis = 1)

print(sparkDF.head())

sparkDF=spark.createDataFrame(df_main)

# add extra age to fill the gap of sep
extract_age = F.udf(lambda x: x[0])
sparkDF = sparkDF.withColumn('age_temp', extract_age('age_on_admittance')).withColumn('age_on_admittance', F.concat(F.col('age_on_admittance'),F.array(F.col('age_temp')))).drop('age_temp')

print(sparkDF.head())

w = Window.partitionBy('subject_id').orderBy('intime')
# sort and merge ccs and age
sparkDF = sparkDF.withColumn('icd_code', F.collect_list('icd_code').over(w)).withColumn('age_on_admittance', F.collect_list('age_on_admittance').over(w)).groupBy('subject_id').agg(F.max('icd_code').alias('icd_code'), F.max('age_on_admittance').alias('age_on_admittance'))

print(sparkDF.head())

sparkDF.write.parquet('behrt_format_mimic4ed')


# diagnoses = EHR(diagnoses).array_flatten(config['col_name']).array_flatten('age')
# diagnoses.write.parquet(config['output'])




















