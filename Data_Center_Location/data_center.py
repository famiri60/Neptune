import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import re

# Load your data into a DataFrame
# Replace 'data.csv' with your actual data file
df = pd.read_csv('Data_Center_Location/combined_data.csv')

# Create a Streamlit app
st.title("Data Center Locations")

# Sidebar for user selection
#display_choice = st.radio("Choose Data Display:", ("Continent", "Company"))
# Sidebar for user selection
with st.sidebar:
    display_choice = st.radio("Choose Data Display:", ("Continent", "Company"))
    if display_choice == "Continent":
        selected_continent = st.selectbox("Select Continent:", df['Continent'].unique())
    else:
        selected_company = st.selectbox("Select Company:", df['Company'].unique())

if display_choice == "Continent":
    #st.title("Data Center Locations")
    #st.sidebar.warning("Selecting 'Continent'")
    #selected_continent = st.selectbox("Select Continent:", df['Continent'].unique())
    filtered_df = df[df['Continent'] == selected_continent]
    # Create a bar plot to show the frequency of data centers for each company
    company_counts = filtered_df['Company'].value_counts().reset_index()
    st.subheader("Data Centers in "+selected_continent.title())
    bar_chart1 = px.bar(company_counts, x=company_counts['Company'], y=company_counts['count'], color_discrete_sequence=['blue'])
    # Set the title for the bar chart
    bar_chart1.update_layout(width=320, height=500, title="Frequency of data centers in "+selected_continent.title())
    list_of_dicts = filtered_df.apply(lambda row: {'Location': f"{row['Location']}", 'Company':f"{row['Company']}"}, axis=1).to_list()

    # Extract unique product names and certification names
    location_names = list(set(item['Location'] for item in list_of_dicts))
    company_names = list(set(item['Company'] for item in list_of_dicts))

    # Create node labels
    node_labels = location_names + company_names

    # Create links between locations and companies
    source_indices = [location_names.index(item['Location']) for item in list_of_dicts]
    target_indices = [len(location_names) + company_names.index(item['Company']) for item in list_of_dicts]
    link_values = [1] * len(list_of_dicts)

    # Create a Sankey diagram
    fig1 = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels
                    ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=link_values
        )
    ))


    fig1.update_layout(width=300, height=600, title_text="Location Company Flow")

    ddf=pd.DataFrame(filtered_df.groupby('Location')['Company'])
    ddf.columns=['Location','Company']
    ddf['Company'] = ddf['Company'].astype(str)
    def preprocess(x):
        return x.split('Name')[0].replace('\n', ' ')
    ddf['Company'] = ddf['Company'].apply(preprocess)

    
    # Extract and concatenate company names
    company_names = ['meta', 'google', 'amazon', 'microsoft', 'alibaba', 'oracle']

    # Function to extract and concatenate company names
    def extract_and_concatenate_companies(description):
        if isinstance(description, str):
            extracted_companies = [company for company in company_names if re.search(r'\b' + company + r'\b', description, re.IGNORECASE)]
            return ', '.join(extracted_companies)
        else:
            return None

    # Apply the extraction function to the 'Company' column
    ddf['Company'] = ddf['Company'].apply(extract_and_concatenate_companies)

    # Create a new DataFrame with the updated 'Company' column
    result_df = pd.DataFrame(ddf)
    result_df=result_df.merge(filtered_df, on='Location', how='left')[['Location', 'Company_x', "Latitude", "Longitude"]]
    result_df.columns=['Location', 'Company', 'Latitude', 'Longitude']
    results = result_df.drop_duplicates().reset_index(drop=True)
    df_no_duplicates = results.drop_duplicates(subset=['Location']).reset_index(drop=True)
    
    # Create a Scatter Mapbox for the selected data
    fig = go.Figure(go.Scattermapbox(
        lat=df_no_duplicates['Latitude'],
        lon=df_no_duplicates['Longitude'],
        mode='markers',
        marker=dict(size=10),
        hovertext=df_no_duplicates.apply(lambda row: '<br>'.join(f'{column}: {value}' for column, value in row.items()), axis=1),
        hoverinfo='text',
    ))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            zoom=1,
        ),
        title='Data centers',
    )
    st.plotly_chart(fig)
    #st.plotly_chart(bar_chart1)
    #st.plotly_chart(fig1)
    # Split the app into two columns
    col1, col2 = st.columns(2)

    # In the first column, put fig and bar_chart1
    with col1:
        #st.plotly_chart(fig)
        st.plotly_chart(bar_chart1)
        
        
       
        
        

    # In the second column, put fig1
    with col2:
        st.plotly_chart(fig1)
         
        
        
    
    
    
else:
    #st.sidebar.warning("Selecting 'Company'")
    #selected_company = st.selectbox("Select Company:", df['Company'].unique())
    filtered_df = df[df['Company'] == selected_company]
    
    # Load the company logo based on the selected company
    logo_image_path = f'images/{selected_company.lower()}_datacenter.png'  # Adjust the path as needed
    col1, col2 = st.columns(2)

    # In the first column, put fig and bar_chart1
    with col1:
        #st.title("Data Center Locations")
        st.subheader("Data Centers of "+selected_company.title())
        
        
       
        
        

    # In the second column, put fig1
    with col2:
      
        # Display the company logo
        st.image(logo_image_path, use_column_width=False)
    # Create a bar plot to show the frequency of data centers for each continent
    continent_counts = filtered_df['Continent'].value_counts().reset_index()
    
    bar_chart1 = px.bar(continent_counts, x=continent_counts['Continent'], y=continent_counts['count'], color_discrete_sequence=['blue'])
    # Set the title for the bar chart
    bar_chart1.update_layout(title="Frequency of data centers of "+ selected_company.title() +" across different continents")
    


    # Create a Scatter Mapbox for the selected data
    fig = go.Figure(go.Scattermapbox(
        lat=filtered_df['Latitude'],
        lon=filtered_df['Longitude'],
        mode='markers',
        marker=dict(size=10),
        hovertext=filtered_df[['Location', "Latitude", "Longitude", 'Zones', 'Opened']].apply(lambda row: '<br>'.join(f'{column}: {value}' for column, value in row.items()), axis=1),
        hoverinfo='text',
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            zoom=1,
        ),
        title='Data centers',
    )


    st.plotly_chart(fig)
    #st.subheader("frequency of data centers of the selected company across different continents")

    st.plotly_chart(bar_chart1)
