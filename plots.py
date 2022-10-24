import numpy as np
import pandas as pd
import random

import plotly.express as px
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
        # title = title,
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
        # title_text="Basic Sankey Diagram",
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

def getDistributionplot(dataset,column_name):
    """ 
    Creates distribution plot.
    """
    fig = px.histogram(dataset, x=column_name, color="Response", marginal="violin", # can be `box`, `violin`
                         hover_data=dataset.columns)
    return fig

def getPairPlot(dataset):
    """
    Creates Pairplot
    """
    fig = sns.pairplot(
        dataset[['Age','Annual_Premium','Vintage','Response']],
        hue="Response"
        )
    return fig

def getRelPlot(dataset,x,y):
    """
    Creates Pairplot
    """
    fig = sns.relplot(data=dataset,x=x,y=y,kind='line',hue='Response')
    return fig

def getCorrelationPlot(dataset):
    """
    Creates Pairplot
    """
    dataset = dataset.astype({'Response':'int'})
    dataset.Gender = dataset.Gender.map({'Male':0,'Female':1})
    dataset.Vehicle_Damage = dataset.Vehicle_Damage.map({'No':0,'Yes':1})
    
    SpearmanCorr = dataset.corr(method="spearman")
    fig = px.imshow(SpearmanCorr,text_auto=True,color_continuous_scale='RdBu_r')
    fig.update_layout(
        autosize=False,
        width=1200,
        height=800
        )
    return fig

def getSunburstPlot(dataset,path):
    """
    Generate Sunburst with the given path
    """
    fig = px.sunburst(dataset, path=path, values='count_',width=750,height=750)
    fig.update_traces(textinfo="label+percent parent")
    return fig

def z_score(df):
    return (df-df.mean())/df.std(ddof=0)

def generate_random_color(n):
    arr = []
    for i in range(n):
        r = lambda: random.randint(0,255)
        hex_number = '#%02X%02X%02X'% (r(),r(),r())
        arr.append(hex_number)
    return arr

def parallel_coordinate_plots_before_and_after_zscore(df):
    """
    This function generates all possbile parallel_coordinate_plots for non interger
    columns before and after standard sclaing side by side.
    
    Parameters
    ----------
    df : Input dataframe
    
    Returns
    -------
    This function returns nothing
    
    """
    number_columns = df.select_dtypes(include=[np.number]).columns.values
    non_number_columns = list(set(df.columns) - set(number_columns))    
    
    df_scaled = df[number_columns].apply(z_score)
    df_scaled[non_number_columns] = df[non_number_columns]
    
        
    for non_number_column in non_number_columns:
        f, (a1,a2) = plt.subplots(ncols=2, nrows=1, figsize=(20,5))
        
        print('Plotting for column "',non_number_column, '" before and after zscore scaling side by side')
        pd.plotting.parallel_coordinates(df[np.append(number_columns,non_number_column)],
                                         class_column=non_number_column,
                                         color=generate_random_color(len(non_number_columns)),
                                         ax=a1)
        a1.set_title('Before Scaling')
        pd.plotting.parallel_coordinates(df_scaled[np.append(number_columns,non_number_column)],
                                         class_column=non_number_column,
                                         ax=a2)
        a2.set_title('After Standard Scaling')

        plt.show()

