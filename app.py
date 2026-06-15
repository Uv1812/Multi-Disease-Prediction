# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 11:30:43 2025

@author: Admin
"""

import pickle
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import os

# ✅ Relative paths — works locally and on Streamlit Cloud
base = os.path.dirname(__file__)

dia_mod = pickle.load(open(os.path.join(base, "model", "diabetes_model.sav"), 'rb'))
ht_mod  = pickle.load(open(os.path.join(base, "model", "heart_model.sav"), 'rb'))
par_mod = pickle.load(open(os.path.join(base, "model", "parkinson_pre.sav"), 'rb'))


with st.sidebar:
    selected=option_menu('Multiplr Disease Prediction System',
                         ['Diabetes Prediction',
                          'Heart Disease Prediction',
                          'Parkinsons Prediction'],
                         icons=['activity','heart','person'],
                         default_index=0)
    
if(selected=='Diabetes Prediction'):
    st.title('Diabetes Prediction Web App')
    
    c1,c2,c3=st.columns(3)
    # Getting input data
    with c1:
       Pregnancies=st.text_input('Number of Pregnancies')
    with c2:
       Glucose=st.text_input('Glucose level')
    with c3:
       BloodPressure=st.text_input('Blood pressure value')
    with c1:   
       SkinThickness=st.text_input('Skin thchness value')
    with c2:   
       Insulin=st.text_input('Insulin level')
    with c3:   
       BMI=st.text_input('BMI Value')
    with c1:  
       DiabetesPedigreeFunction=st.text_input('Diabetes Pedigree Function value')
    with c2:   
       Age=st.text_input('Age of person')
   
    # code for Prediction
    diagnosis=''
    if st.button('Diabetes test result'):
        data=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
        pred= dia_mod.predict(np.asarray(data).reshape(1,-1))
    
        if(pred[0]==1):
            diagnosis='The person has Diabetic'
        else:
            diagnosis='The Person is not Diabetic'
    st.success(diagnosis)
    
if(selected=='Heart Disease Prediction'):
    st.title('Heart Disease Prediction')
    
    c1,c2,c3=st.columns(3)
    # Getting input data
    with c1:
       age=st.text_input('Age of person')
    with c2:
       sex=st.text_input('Sex')
    with c3:
       cp=st.text_input('Cp')
    with c1:   
       trestbps=st.text_input('trestbps')
    with c2:   
       chol=st.text_input('chol')
    with c3:   
       fbs=st.text_input('fbs')
    with c1:  
       restecg=st.text_input('restecg')
    with c2:   
       thalach=st.text_input('thalach')
    with c3:
       exang=st.text_input('exang')
    with c1:   
       oldpeak=st.text_input('oldpeak')
    with c2:   
       slope=st.text_input('slope')
    with c3:   
       ca=st.text_input('ca')
    with c1:  
       thal=st.text_input('thal')
   
    # code for Prediction
    diagnosis=''
    if st.button('Heart test result'):
        data=[float(age),float(sex),float(cp),float(trestbps),float(chol),float(fbs),float(restecg),float(thalach),float(exang),float(oldpeak),float(slope),float(ca),float(thal)]
        pred= ht_mod.predict(np.asarray(data).reshape(1,-1))
    
        if(pred[0]==1):
            diagnosis='The person has Heart disease'
        else:
            diagnosis='The person has no Heart disease'
    st.success(diagnosis)

if(selected=='Parkinsons Prediction'):
    st.title('Parkinsons Disease Prediction')
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
   
    # code for Prediction
    diagnosis=''
    if st.button('Parkinson test result'):
        pred = par_mod.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
    
        if(pred[0]==1):
            diagnosis="The person has Parkinson's  disease"
        else:
            diagnosis="The person has no Parkinson's  disease"
    st.success(diagnosis)
