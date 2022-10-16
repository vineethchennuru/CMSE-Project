import random
import pandas as pd
import numpy as np

import plotly
import plotly.graph_objects as go
from pyrsistent import thaw

import streamlit as st
st.set_page_config(layout="wide")

# Reading data
original_data = pd.read_csv('Health_Insurance_Cross_Sell_Prediction.csv')
data = original_data.copy()

data['count_'] = 1
data.drop('id',axis=1,inplace=True)


data.Driving_License = data.Driving_License.map({1:'YES',0:'NO'})

criteria = [data.Age.between(20, 39), data.Age.between(40, 50), data.Age.between(51, 85)]
values = ['20-40','41-60','61-85']
data.Age = np.select(criteria, values, 0)

criteria = [data.Region_Code.between(0, 18), data.Region_Code.between(19, 36), data.Region_Code.between(37, 52)]
values = ['0-18','19-36','37-52']
data.Region_Code = np.select(criteria, values, 0)


def generate_random_color(n):
    arr = []
    for i in range(n):
        r = lambda: random.randint(0,255)
        hex_number = '#%02X%02X%02X'% (r(),r(),r())
        arr.append(hex_number)
    return arr


def genSankey(df,cat_cols=[],value_cols='',title='Sankey Diagram'):
    # maximum of 6 value cols -> 6 colors
    # colorPalette = ['#4B8BBE','#306998','#FFE873','#FFD43B','#646464']
    colorPalette = generate_random_color(len(cat_cols))
    labelList = []
    colorNumList = []
    for catCol in cat_cols:
        labelListTemp =  list(set(df[catCol].values))
        colorNumList.append(len(labelListTemp))
        labelList = labelList + labelListTemp
        
    # remove duplicates from labelList
    labelList = list(dict.fromkeys(labelList))
    
    # define colors based on number of levels
    colorList = []
    for idx, colorNum in enumerate(colorNumList):
        colorList = colorList + [colorPalette[idx]]*colorNum
        
    # transform df into a source-target pair
    for i in range(len(cat_cols)-1):
        if i==0:
            sourceTargetDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
            sourceTargetDf.columns = ['source','target','count']
        else:
            tempDf = df[[cat_cols[i],cat_cols[i+1],value_cols]]
            tempDf.columns = ['source','target','count']
            sourceTargetDf = pd.concat([sourceTargetDf,tempDf])
            
        sourceTargetDf = sourceTargetDf.groupby(['source','target']).agg({'count':'sum'}).reset_index()
        
        
    # add index for source-target pair
    sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(lambda x: labelList.index(x))
    sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(lambda x: labelList.index(x))
    
    
    # creating the sankey diagram
    data = dict(
        type='sankey',
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(
            color = "black",
            width = 0.5
          ),
          label = labelList,
          color = colorList
        ),
        link = dict(
          source = sourceTargetDf['sourceID'],
          target = sourceTargetDf['targetID'],
          value = sourceTargetDf['count']
        )
      )
    
    layout =  dict(
        title = title,
        font = dict(
          size = 10
        )
    )
       
    fig_dict = dict(data=[data], layout=layout)
    fig = go.Figure(data=fig_dict['data'],layout=fig_dict['layout'])

    for x_coordinate, column_name in enumerate(cat_cols):
      fig.add_annotation(
          x=x_coordinate,
          y=1.05,
          xref="x",
          yref="paper",
          text=column_name,
          showarrow=False,
          font=dict(
              family="Courier New, monospace",
              size=16,
              color="black"
              ),
          align="center"
      )

    fig.update_layout(
        title_text="Basic Sankey Diagram", 
        xaxis={
        'showgrid': False, # thin lines in the background
        'zeroline': False, # thick line at x=0
        'visible': False,  # numbers below
        },
        yaxis={
        'showgrid': False, # thin lines in the background
        'zeroline': False, # thick line at x=0
        'visible': False,  # numbers below
        },
        plot_bgcolor='rgba(0,0,0,0)',
        font_size=10
    )

    return fig

cols = ['Gender','Age','Region_Code','Vehicle_Age','Vehicle_Damage','Response']
fig = genSankey(data,cat_cols=cols,value_cols='count_',title='Sankey')

# Plot!
st.plotly_chart(fig, use_container_width=True)
