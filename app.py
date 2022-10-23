from pickle import TRUE
import random
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st

from plots import *

st.set_page_config(page_title='CMSE Project',layout="wide")

st.markdown("<h1 style='text-align: center; color: white;'>CMSE Project Webapp</h1>", unsafe_allow_html=True)

st.markdown(
    '<p><center><img alt="Car Insurance" src="https://www.financialexpress.com/wp-content/uploads/2022/10/Why-you-should-buy-your-new-car-insurance-directly-from-the-insurer_Reference-image.png"  width="400" height="200"> </center></p>', unsafe_allow_html=True
    )

st.markdown('<p><hr></p>',unsafe_allow_html=True)

st.markdown(
    '<p style="color:magenta; font-size:22px"><u>Abstract</u>: Predict Health Insurance Owners who will be interested in Vehicle Insurance',unsafe_allow_html=True
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
data = original_data.copy()

data['count_'] = 1


data.Driving_License = data.Driving_License.map({1:'YES',0:'NO'})

criteria = [data.Age.between(20, 39), data.Age.between(40, 50), data.Age.between(51, 85)]
values = ['20-40','41-60','61-85']
data.Age = np.select(criteria, values, 0)

criteria = [data.Region_Code.between(0, 18), data.Region_Code.between(19, 36), data.Region_Code.between(37, 52)]
values = ['0-18','19-36','37-52']
data.Region_Code = np.select(criteria, values, 0)

st.header('Sample Data')
st.table(original_data.iloc[1:].head(10))

cols = ['Gender','Age','Region_Code','Vehicle_Age','Vehicle_Damage','Response']
fig = genSankey(data,cat_cols=cols,value_cols='count_',title='Sankey')


st.header('Sankey chart with few columns of the dataset')
# Plot!
st.plotly_chart(fig, use_container_width=True)


tab1, tab2, tab3 = st.tabs(["Basic Analysis-1", "Basic Analysis-2", "Plots for next steps"])

with tab1:
    st.header("Basic Analysis-1")

    col1, col2 = st.columns(2,gap='large')
    with col1:
        options = st.selectbox('Select Column',( 'Age', 'Region_Code','Policy_Sales_Channel', 'Vintage'))
        fig = boxPlotter(original_data, options)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.markdown("Boxplots for "+options+" as parameter: ")
        st.pyplot(fig)

    # with col2:
    #     column_name =  st.selectbox(
    #         'For which column would you like to see the distplot',
    #         df.columns[:-1])

    #     # st.write('You selected:', column_name)


    #     plt.figure(figsize=(20, 15))

    #     fig = sns.displot(df[column_name])

    #     st.header("Simple Distplot")
    #     st.pyplot(fig)


    
with tab2:
    st.header("Basic Analysis-2")
    col1, col2 = st.columns(2,gap='large')

    with col1:

        options = st.selectbox('Hue for the plot',('Gender', 'Vehicle_Age', 'Vehicle_Damage'))

        st.set_option('deprecation.showPyplotGlobalUse', False)
        # st.write('You selected:', options)

        fig = sns.relplot(data=original_data,x='Age',y='Previously_Insured',kind='line',hue=options)
        st.markdown("Relation plot of Age vs Previously_Insured with hue: "+options)
        st.pyplot(fig)

    # with col2:
    #     column_name =  st.selectbox(
    #         'For which column would you like to see the distplot',
    #         df.columns[:-1])

    #     # st.write('You selected:', column_name)


    #     plt.figure(figsize=(20, 15))

    #     fig = sns.displot(df[column_name])

    #     st.header("Simple Distplot")
    #     st.pyplot(fig)


with tab3:
    st.header("Basic Analysis-3")
    st.markdown('Correaltion matrix for the dataset')
    SpearmanCorr = original_data.corr(method="spearman")
    fig = px.imshow(SpearmanCorr,text_auto=True,color_continuous_scale='RdBu_r')
    fig.update_layout(
        autosize=False,
        width=800,
        height=800)

    st.plotly_chart(fig)

