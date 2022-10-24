from pickle import TRUE
import random
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import hiplot as hip

import streamlit as st

from plots import *

st.set_page_config(page_title='CMSE Project',layout="wide")

st.markdown("<h1 style=' text-align: center; color: white;'>CMSE Project Webapp</h1>", unsafe_allow_html=True)

st.markdown(
    '<p><center><img alt="Car Insurance" src="https://www.financialexpress.com/wp-content/uploads/2022/10/Why-you-should-buy-your-new-car-insurance-directly-from-the-insurer_Reference-image.png"  width="500" height="250"> </center></p>', unsafe_allow_html=True
    )

st.markdown('<p><hr></p>',unsafe_allow_html=True)

st.markdown(
    '<p style="color:magenta; font-size:22px"><u>Abstract</u>: Predict Health Insurance Owners who will be interested in Vehicle Insurance (Classification Problem)',unsafe_allow_html=True
)
st.markdown(
    '<p><u>Part1</u>: I will be exploring the dataset(understanding all the given columns), performing different EDA techniques, and deploying a dashboard with various visualization to slice, dice, and generate valuable insights from the data."',unsafe_allow_html=True
)

st.markdown(
    '<p ><u>Part2</u>: Taking insights generated from Part1 and building a model to predict whether the policyholders (customers) from the past year will also be interested in Vehicle Insurance provided by the company."',unsafe_allow_html=True
)

st.markdown('<p><hr></p>',unsafe_allow_html=True)

st.markdown('Code Repository : [GitHub Code Repo](https://github.com/vineethchennuru/CMSE-Project)')
st.markdown('Dataset link : [Health Insurance Cross Sell Prediction Dataset🏠🏥](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)')

st.markdown('<p><hr></p>',unsafe_allow_html=True)


# Reading data
original_data = pd.read_csv('Health_Insurance_Cross_Sell_Prediction.csv')

original_data.drop('id',axis=1,inplace=True)

original_data = original_data.astype({'Response':'string'})
original_data = original_data.astype({'Region_Code':'int'})
original_data = original_data.astype({'Annual_Premium':'int'})
original_data = original_data.astype({'Policy_Sales_Channel':'int'})

# For computational purposes taking 20% of data for analysis
# original_data = original_data.sample(n=int(len(original_data)/20))
original_data = original_data[0:int(len(original_data)/20)]
data = original_data.copy()

data['count_'] = 1

data.Driving_License = data.Driving_License.map({1:'YES',0:'NO'})

criteria = [data.Age.between(20, 39), data.Age.between(40, 50), data.Age.between(51, 85)]
values = ['20-40','41-60','61-85']
data.Age = np.select(criteria, values, 0)

criteria = [data.Region_Code.between(0, 18), data.Region_Code.between(19, 36), data.Region_Code.between(37, 52)]
values = ['0-18','19-36','37-52']
data.Region_Code = np.select(criteria, values, 0)

# CSS to inject contained in a string
hide_table_row_index = """
            <style>

            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)

st.header('Sample Data')
st.table(original_data.iloc[1:].head(10))

st.header('Data Description')
st.table(pd.read_csv('data_description.csv'))

st.markdown('<p><hr></p>',unsafe_allow_html=True)

cols = ['Gender','Age','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Response']
st.header('Sankey chart')
options = st.multiselect(
    'Select columns you want to visualize in Sankey chart',
    cols,
    cols)
fig = genSankey(data,cat_cols=options,value_cols='count_',title='Sankey')
# Plot!
st.plotly_chart(fig, use_container_width=True)

tab1, tab2, tab3 = st.tabs(["1D Analysis", "2D Analysis", "Plots and steps for next steps"])

with tab1:
    st.header("One dimentional analysis of different columns w.r.t to Response")

    # Plot-1 in Tab-1
    column_name =  st.selectbox(
        'For which column would you like to see the distplot',
        ('Age','Region_Code','Annual_Premium','Policy_Sales_Channel','Vintage'))
    fig = getDistributionplot(original_data,column_name)
    st.markdown("Distplot for "+column_name)
    st.plotly_chart(fig,use_container_width = True)

    # Plot-2 in Tab-1
    path = ['Gender','Age','Region_Code','Previously_Insured','Vehicle_Age','Vehicle_Damage','Response']
    options = st.multiselect(
        'Select columns you want to visualize in Sunburst chart',
        path,
        ['Region_Code','Age','Response'])
    if len(options) >4:
        st.markdown(
            '<p style="color:red; font-size:22px">For better understanding purpose please select less than or equal to 4 columns in the filter</p>',unsafe_allow_html=True)
    else:
        fig = getSunburstPlot(data,options)
        st.markdown('Sunburst chart for given path')
        st.plotly_chart(fig,use_container_width = True)

with tab2:
    st.header("Two dimentional analysis of different columns w.r.t to Response")
    col1, col2 = st.columns(2,gap='large')

    with col1:
        options_cols_x = ('Age', 'Driving_License', 'Region_Code', 'Previously_Insured')
        options_cols_y = ('Annual_Premium', 'Policy_Sales_Channel', 'Vintage')

        options_x = st.selectbox('x-axis',options_cols_x)
        options_y = st.selectbox('y-axis',options_cols_y)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        # st.write('You selected:', options)

        fig = getRelPlot(original_data,options_x,options_y)
        st.markdown("Relation plot of "+ options_x+ " vs "+ options_y +" with hue: Response")
        st.pyplot(fig,use_container_width = True)

    with col2:
        st.markdown('<p><br></p>',unsafe_allow_html=True)
        st.markdown('<p><br></p>',unsafe_allow_html=True)
        st.markdown('<p><br></p>',unsafe_allow_html=True)
        fig = getPairPlot(original_data)
        st.markdown("Pairplot")
        st.pyplot(fig,use_container_width = True)


with tab3:
    st.header("Plots and insights that will help us for modelling")
    st.markdown('Correaltion matrix for the dataset')
    fig = getCorrelationPlot(original_data)
    st.plotly_chart(fig)


    st.markdown('Few intresting stats about the data')
    original_data = original_data.astype({'Response':'int'})
    df_descibe = original_data.describe().reset_index().rename({'index':''},axis=1)
    st.table(df_descibe)


st.markdown('<p><hr></p>',unsafe_allow_html=True)

xp = hip.Experiment.from_dataframe(original_data)

# Instead of calling directly `.display()`
# just convert it to a streamlit component with `.to_streamlit()` before
st.markdown('HiPlot of dataset')
ret_val = xp.to_streamlit().display()

st.subheader('Next we will be trying to do preprocessing on data and modelling the data with different models')