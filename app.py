import pickle
import streamlit as st
import pandas as pd
import sklearn 
import numpy as np

lr = pickle.load(open('lr1.pkl', 'rb'))  # rb: read bianry
dt = pickle.load(open('dt1.pkl', 'rb'))  # rb: read bianry
rf = pickle.load(open('rf1.pkl', 'rb'))  # rb: read bianry

model= st.sidebar.selectbox('Select the model', ['LogReg', 'Decision Tree', 'Random Forest'])
 

st.title('ME/CFS, Depression Predictor')

st.write('Fill the below details for a predictive diagnosis')

col1,col2= st.columns(2)


with col1:
  age= st.number_input('Enter Age', 18,70,25)
  gender= st.selectbox('Enter Gender', ('Male', 'Female'))
  sq_index= st.number_input('Sleep Quality Index', 1.0,10.0, 5.0)
  bf_level= st.number_input('Brain Fog Level', 1.0,10.0,5.0)
  pain_score= st.slider('Physical Pain Score', 1.0,10.0, 5.0)
  stress_level= st.slider('Stress Level', 1.0,10.0, 5.0)
  phq9= st.number_input('Depression PHQ9 score', 0,27,15)
  fat_sev= st.slider('Fatigue Severity Scale Score', 0,10,5)
with col2:
  pem_duration= st.number_input('PEM Duration', 0, 50, 0)
  sleep= st.number_input('Hours of sleep per night', 0,20,0)
  pem_present= st.selectbox('PEM present',['Yes', 'No'])
  work_status= st.selectbox('Work Status', ('Working','Partially Working','Not Working'))
  social_act= st.selectbox('Social Activity Level', ('Very Low', 'Low', 'Medium', 'High', 'Very High' ))
  ex_f= st.selectbox('exercise_frequency', ('Daily', 'Often', 'Rarely', 'Never', 'Sometimes'))
  med_mind= st.selectbox('meditation_or_mindfulness', ('Yes', 'No'))


if gender=='Male':
  gen_m=1
  gen_f=0
else:
  gen_m=0
  gen_f=1

if pem_present=='Yes':
  pem=1
else:
  pem=0

if work_status=='Working':
  ws_work=2
elif work_status=='Partially Working':
  ws_work=1
else:
  ws_work=0

if social_act=="Very Low":
  sa=1
elif social_act=="Low":
  sa=2
elif social_act=="Medium":
  sa=3
elif social_act=="High":
  sa=4
elif social_act=="Very High":
  sa=5

if ex_f=="Daily":
  exf=4
elif ex_f=="Often":
  exf=3
elif ex_f=="Rarely":
  exf=1
elif ex_f=="Never":
  exf=0
elif ex_f=="Sometimes":
  exf=2

if med_mind=="Yes":
  mm=1
else:
  mm=0

test_input= [[age, gen_m, sq_index, bf_level, pain_score, stress_level, phq9, 
             fat_sev, pem_duration, sleep, pem, ws_work, 
             sa, exf, mm]]

df_columns= [['age', 'gender', 'sleep_quality_index', 'brain_fog_level',
             'physical_pain_score', 'stress_level', 'depression_phq9_score',
            'fatigue_severity_scale_score', 'pem_duration_hours',
            'hours_of_sleep_per_night', 'pem_present', 'work_status',
            'social_activity_level', 'exercise_frequency',
            'meditation_or_mindfulness']]

test_df= pd.DataFrame(test_input, columns=df_columns)
st.write('Test Data')
st.write(test_df)


if st.button('Predict'):
    if model=='LogReg':
        pred= lr.predict(test_df)
        st.success(pred[0])
    elif model=='Decision Tree':
        pred=dt.predict(test_df)
        st.success(pred[0])
    else:
        pred=rf.predict(test_df)
        st.success(pred[0])



