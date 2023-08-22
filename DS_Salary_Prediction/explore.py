import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

@st.cache
def load_data():
    df = pd.read_csv("DS_Salary_Prediction/clean_data3.csv")
    return df

def explore_page():
    st.title("Explore Data Scientist Salaries")
    st.write("""### Data scientist salary EDA 2021""")
    data=load_data()
    job_percent=(data['Job Location'].value_counts()/len(data['Job Location']))*100
    df_location0=pd.DataFrame(job_percent).reset_index(drop=False)
    df_location0.columns=['job_location', 'percentage']
    df_location=pd.merge(df_location0, data, left_on='job_location', right_on='Job Location', how='left')
    df_location=df_location[['job_location', 'percentage', 'latitude', 'longitude', 'Name']].drop_duplicates().reset_index(drop=True)
    fig1 = px.choropleth(
        df_location,
        locations="job_location",
        locationmode="USA-states",
        color="percentage",
        color_continuous_scale="YlGnBu",  # You can choose other color scales
        scope="usa",
        title="Data Science Job Percentage by States")
    fig1.update_geos(
        visible=True,
        showcoastlines=True,
        coastlinecolor="Black",
        showland=True,
        landcolor="white",
        showlakes=True,
        lakecolor="white"
    )
    st.plotly_chart(fig1)
    col1, col2 = st.columns(2)
    with col2:
        df_sector=pd.DataFrame((data['Sector'].value_counts()/len(data['Sector']))*100).reset_index(drop=False)
        df_sector.columns=['Sector', 'Percentage']
        fig2 = px.bar(df_sector, x='Percentage', y='Sector', orientation='h', title='Data Science Job Persentage by Sector')
        st.plotly_chart(fig2)
    with col1:
        skills = ['Python', 'spark', 'aws', 'excel', 'sql', 'sas', 'keras', 'pytorch',
                'scikit', 'tensor', 'hadoop', 'tableau', 'bi', 'flink', 'mongo', 'google_an']
        skills_by_Sector = data.groupby('Sector')[skills].sum()
        styled_skills = skills_by_Sector.style.background_gradient(cmap='YlGnBu', axis='columns')
        # Display the styled DataFrame in Streamlit
        st.markdown('<h2 style="font-size:16px;"><b>Skills by Job Sector</b></h2>', unsafe_allow_html=True)


        #st.title('Skills by Job Sector')
        st.dataframe(styled_skills)
