from datetime import date
import pandas as pd
import numpy as np
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

#Create PySpark SparkSession
spark = SparkSession.builder \
    .master("local[1]") \
    .appName("SparkByExamples.com") \
    .getOrCreate()



def calculate_age_on_current_admission(admission_date,anchor_time,anchor_age):
    age = admission_date.year - anchor_time.year - ((admission_date.month, admission_date.day) < (anchor_time.month, anchor_time.day)) + anchor_age
    return age  

df = pd.read_pickle('df_main.pkl')
df['T'] = df.groupby('subject_id')['subject_id'].transform('count')

df['T'] = df.groupby('subject_id')['subject_id'].transform('count')


print(df.head())

id_amount = len(pd.unique(df['subject_id']))

print(id_amount)

df = df[(df['T'] >= 5)]

print(df.head())

qualified_patients = len(pd.unique(df['subject_id']))

print(qualified_patients)




