import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image

@st.cache_data
def load_data():
    df = pd.read_csv("./clean_data3.csv")
    return df
df=load_data()
# Load the saved model and other data
def load_model():
    with open('saved_steps1.pkl', 'rb') as file:
        data = pickle.load(file)
    return data
data=load_model()

# Load the models and label encoders from the loaded data
regressor_loaded0 = data["model0"]
#regressor_loaded1 = data["model1"]
#regressor_loaded2 = data["model2"]
le_location = data["le_location"]
le_size = data["le_size"]
le_sector = data["le_sector"]
le_seniority = data["le_seniority"]
le_degree = data["le_degree"]

# Function to add a background image from a local file
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Function to create the estimation page
def estimation_page():
    title = '<p style="font-family:sans-serif; color:Purple; font-size: 45px">Data Scientist Salary Estimation</p>'
    st.markdown(title, unsafe_allow_html=True)
    
    image = Image.open('data-science.jpg')
    st.image(image, caption='Image: Wright Studio/Shutterstock [Reference: https://www.techrepublic.com/article/cheat-sheet-how-to-become-a-data-scientist/]')
    
     
    # Display sub-title and input fields for user data
    sub_title = '<p style="font-family:sans-serif; color:Purple; font-size: 25px">Please provide all the required informations</p>'
    st.markdown(sub_title, unsafe_allow_html=True)
    

    # Lists of options for selection
    Location=(sorted(df['Job Location'].unique()))
    
       
    Company_Size = ('Large', 'Medium', 'Small')

    Industry_Sector = (sorted(df['Sector'].unique()))

    
    Level = ('Junior', 'Sinior')
    
    Degree = ('Master', 'PhD')
    
    # Create a form to gather user input
    with st.form("my_form",clear_on_submit=True):

        location=st.selectbox('Location', Location)

        size=st.selectbox("Company Size",Company_Size)

        sector=st.selectbox("Industry Sector", Industry_Sector)

        seniority=st.selectbox("Level", Level)
        if seniority=="Junior":
            seniority='jr'
        else:
            seniority='sr'

        degree=st.selectbox("Degree", Degree)

        rating=st.slider("Rating", 1, 5, 1)


 
        sub_title2 = '<p style="font-family:sans-serif; color:Purple; font-size: 25px">Please select your skills</p>'
        st.markdown(sub_title2, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:

            # Create checkboxes for skills
            python=st.checkbox("Python")
            if python:
                python=1.
            else:
                python=0.


            #spark=st.slider("Spark", 0., 1., 0.5)
            spark=st.checkbox("Spark")
            if spark:
                spark=1.
            else:
                spark=0.

            #aws=st.slider("AWS", 0., 1., 0.5)
            aws=st.checkbox("AWS")
            if aws:
                aws=1.
            else:
                aws=0.

            #excel=st.slider("Excel", 0., 1., 0.5)
            excel=st.checkbox("Excel")
            if excel:
                excel=1.
            else:
                excel=0.

            #sql=st.slider("SQL", 0., 1., 0.5)
            sql=st.checkbox("SQL")
            if sql:
                sql=1.
            else:
                sql=0.

            #sas=st.slider("SAS", 0., 1., 0.5)
            sas=st.checkbox("SAS")
            if sas:
                sas=1.
            else:
                sas=0.
            #keras=st.slider("Keras", 0., 1., 0.5)
            keras=st.checkbox("Keras")
            if keras:
                keras=1.
            else:
                keras=0.

            pytorch=st.checkbox("Pytorch")
            if pytorch:
                pytorch=1.
            else:
                pytorch=0.


        with col2:
            
            scikit=st.checkbox("Scikit")
            if scikit:
                scikit=1.
            else:
                scikit=0.
            
            tensor=st.checkbox("Tensor")
            if tensor:
                tensor=1.
            else:
                tensor=0.
            
            hadoop=st.checkbox("Hadoop")
            if hadoop:
                hadoop=1.
            else:
                hadoop=0.
           
            tableau=st.checkbox("Tableau")
            if tableau:
                tableau=1.
            else:
                tableau=0.
            
            bi=st.checkbox("BI")
            if bi:
                bi=1.
            else:
                bi=0.
            
            flink=st.checkbox("Flink")
            if flink:
                flink=1.
            else:
                flink=0.
           
            mongo=st.checkbox("Mongo")
            if mongo:
                mongo=1.
            else:
                mongo=0.
           
            google_an=st.checkbox("Google_an")
            if google_an:
                google_an=1.
            else:
                google_an=0.


        # Create a button to submit the form
        ok=st.form_submit_button("Estimate Salary")
        # Process the form submission
        if ok:
            if python==spark==aws==excel==sql==sas==keras==pytorch==scikit==tensor==hadoop==tableau==bi==flink==mongo==google_an==0:
                st.subheader("Please select your skills")


            else:
                st.write("Location:", location, ';', "Company Size:", size, ';', "Industry Sector:", sector, ';', "Company Rating:", str(rating), ';', "level:", seniority, ';', "Degree:", degree)
                skill=[]
                skill_dic={'Python':python, 'Spark':spark, 'AWS':aws, 'Excel':excel, 'SQL':sql, 'SAS':sas, 'Keras':keras,'Pytorch':pytorch,'Scikit':scikit, 'Tensor':tensor, 'Hadoop':hadoop, 'Tableau':tableau, 'BI': bi, 'Flink':                                    flink, 'Mango':mongo, 'Google_an': google_an}
                for s in skill_dic.keys():
                    if skill_dic[s]==1:
                        skill.append(s)
                # Concatenate the list elements into a single string
                list_string = ", ".join(skill)
                st.write("skill(s):", list_string)

                #for i in skill:
                    #st.write(i)
                X = np.array([[rating, location, size, sector, python, spark, aws, excel, sql, sas, keras, pytorch, scikit,
                 tensor, hadoop, tableau, bi, flink, mongo, google_an,seniority, degree]])

                X[:, 1] = le_location.transform(X[:,1])
                X[:, 2] = le_size.transform(X[:,2])
                X[:, 3] = le_sector.transform(X[:,3])
                X[:, 20] = le_seniority.transform(X[:,20])
                X[:, 21] = le_degree.transform(X[:,21])
                X = X.astype(float)
                avg_salary=regressor_loaded0.predict(X)
                st.subheader(f"The estimated average salary is: ${avg_salary[0]:.2f}")
                #if (seniority=='jr') & (degree=='Master') | (avg_salary[0]==min_salary[0]==max_salary[0]):
                    #avg_salary=regressor_loaded0.predict(X)
                    #st.subheader(f"The estimated average salary is: ${avg_salary[0]:.2f}")
                #else:

                    #avg_salary=regressor_loaded0.predict(X)
                    #min_salary=regressor_loaded1.predict(X)
                    #max_salary=regressor_loaded2.predict(X)

                    #if avg_salary[0]>max_salary[0]:
                        #st.subheader(f"The estimated average salary is: ${max_salary[0]:.2f}")

                        #st.subheader(f"The estimated minimum salary is: ${min_salary[0]:.2f}")

                        #st.subheader(f"The estimated maximum salary is: ${avg_salary[0]:.2f}")
                    #else:
                        #st.subheader(f"The estimated average salary is: ${avg_salary[0]:.2f}")

                        #st.subheader(f"The estimated minimum salary is: ${min_salary[0]:.2f}")

                        #st.subheader(f"The estimated maximum salary is: ${max_salary[0]:.2f}")

                    #if avg_salary[0]==min_salary[0]==max_salary[0]:
                        #st.subheader(f"The estimated average salary is: ${avg_salary[0]:.2f}")

        #hit = st.form_submit_button("start over")
    # Create a button to reset the form
    hit=st.button("Reset")
    if hit:
        st.experimental_rerun()
    