import pandas as pd
import random

import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


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
              color="white"
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

def plot_bar(x,data):
    fig = plt.figure(figsize=(10, 4))
    fig = sns.countplot(x= data[x])
    return fig

def boxPlotter(dataset, columnName):
    """
    Plots boxplots for column given as parameter.
    """
    sns.catplot(x="Response", y=columnName, data=dataset, kind="box");
