import streamlit as st
from predict import estimation_page
from explore import explore_page
from predict import add_bg_from_local

page = st.sidebar.selectbox("Predict or Explore", ("Predict", "Explore"))




if page=="Predict":
    estimation_page()
    add_bg_from_local('DS_Salary_Prediction/background.jpg')
else:
    explore_page()
    
