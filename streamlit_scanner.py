#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 20:29:57 2020

@author: tommo
"""


import streamlit as st
from streamlit_folium import folium_static
import folium
import geopandas as gpd
import pandas as pd

from streamlit_folium import folium_static
import folium
from pyproj import Transformer
transformer = Transformer.from_crs("epsg:27700", "epsg:4326")

st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 800px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)

st.write(
     """
#     SCANNER survey
#     """
)
#with st.echo():

@st.cache
def load_data():
    #gdf_gullies = gpd.read_file('Gulleys Nov 2019/gulleys.shp')
    #gdf_gullies.crs = "EPSG:27700"
    #def transform_coords(X1,Y1):
    #    return transformer.transform(X1, Y1)
    
    #gdf_gullies.loc[:,'X1'] = gdf_gullies.apply(lambda x: transform_coords(x['POINT_X'],x['POINT_Y'])[0], axis=1)
    #gdf_gullies.loc[:,'Y1'] = gdf_gullies.apply(lambda x: transform_coords(x['POINT_X'],x['POINT_Y'])[1], axis=1)
    
    
    #gdf_gullies.head()
    df = pd.read_parquet('scannerdata.parquet')
    #df = df.sort_values(['SECTIONLABEL','LABEL','STARTCH'])
    return df


df = load_data()



y = st.sidebar.selectbox("Road:", df['roadcode'].unique(), index=42)

df2 = df[df['roadcode']==y]
selected_chainage = st.slider('Chainage in m', int(df2['cumlength'].min()), int(df2['cumlength'].max()),  \
                              value=(min(11670, max(0,int(df2['cumlength'].max()-1000))),min(17000, int(df2['cumlength'].max()-50))), step=10)
st.write('Selected chainage:', selected_chainage)

params = df.columns[7:49]
with open('Scanner parameters.txt','r') as f:
    available_params = f.readlines()
    available_params = [x.strip() for x in available_params] 
default_selected = [available_params[2],available_params[3],available_params[12],available_params[14],available_params[23],available_params[36]]
params_SELECTED = st.sidebar.multiselect('Select parameters', available_params, default=default_selected)#params)
smoothing = st.sidebar.slider('Smoothing',0,20,(0))



df3 = df2[(df2['SECTIONLABEL'] == 'CL1') & (df2['cumlength'] >= selected_chainage[0]) & (df2['cumlength'] <= selected_chainage[1])]
df4 = df2[(df2['SECTIONLABEL'] == 'CR1') & (df2['cumlength'] >= selected_chainage[0]) & (df2['cumlength'] <= selected_chainage[1])]


import folium

feature_group0 = folium.FeatureGroup(name='Left lane')
feature_group1 = folium.FeatureGroup(name='Right lane')

if df3.shape[0]:
    new_coords = [(df3.X1.min()+df3.X1.max())/2, (df3.Y1.min()+df3.Y1.max())/2]
    hier = df3['Class'].iloc[0]
elif df4.shape[0]:
    new_coords = [(df4.X1.min()+df4.X1.max())/2, (df4.Y1.min()+df4.Y1.max())/2]
    hier = df4['Class'].iloc[0]

#new_coords = transformer.transform((coords[0]+coords[2])/2,  (coords[1]+coords[3])/2)
#def transform_coords(X1,Y1):
#    return transformer.transform(X1, Y1)

mapa = folium.Map(location=new_coords, tiles="Cartodb Positron",
                  zoom_start=12, prefer_canvas=True)


feature_group2 = folium.FeatureGroup(name='Gullies at recommended spacing', show=False)
def plotDot(point):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    #folium.CircleMarker(location=[point.Y1, point.X1],
    #                    radius=3,
    #                    weight=1).add_to(mapa)
    #folium.Marker([point['X1'], point['Y1']],
    #      #Make color/style changes here
    #      icon = folium.simple_marker(color='lightgray', marker_icon='oil'),
    #      ).add_to(mapa)
    color_map = {'CL1':'blue','CR1':'green'}
    
    folium.Circle( [point['X1'], point['Y1']], radius=2
                     , color=color_map[point['SECTIONLABEL']]
                     , fill_color='lightgray'
                     , fill=True
                     ).add_to(feature_group2)
    
feature_group3 = folium.FeatureGroup(name='Actual gullies', show=False)
def plotGul(point):
    folium.Circle( [point['X1'], point['Y1']], radius=2
                     , color='darkgray'
                     , fill_color='black'
                     , fill=True
                     ).add_to(feature_group3)

feature_group5 = folium.FeatureGroup(name='Area of interest', show=True)
def plotDot(point,color):
    folium.Circle( [point['X1'], point['Y1']], radius=2
                     , color=color
                     , fill_color='black'
                     , fill=True
                     ).add_to(feature_group5)
       
    
feature_group4 = folium.FeatureGroup(name='Chainages', show=True)
def plotChain(point):
    #iframe = folium.IFrame(text, width=700, height=450)
    #popup = folium.Popup(iframe, max_width=3000)
    folium.Marker( [point['X1'], point['Y1']], radius=4
                     , color='black'
                     #, fill_color='#808080'
                     #, fill=True
                     , icon=folium.DivIcon(html=str("<p style='font-family:verdana;color:#444;font-size:10px;'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;%d</p>" % (point['cumlength'])))#, point['LABEL'], point['STARTCH'])))
                     #, popup=str(point['cumlength'])
                     ).add_to(feature_group4)
    
#use df.apply(,axis=1) to "iterate" through every row in your dataframe
#df2[df2['gullymarker'] ==1].apply(lambda x: plotDot(x), axis = 1)

df2.iloc[1::15].apply(lambda x: plotChain(x), axis = 1)

df3.iloc[1::5].apply(lambda x: plotDot(x,'red'), axis = 1)
df4.iloc[1::5].apply(lambda x: plotDot(x, 'green'), axis = 1)
#if df3.shape[0] > df4.shape[0]:
#    df3.iloc[1::10].apply(lambda x: plotChain(x), axis = 1)
#else:
#    df4.iloc[1::10].apply(lambda x: plotChain(x), axis = 1)

#gdf_gullies.apply(lambda x: plotGul(x), axis = 1)

#mapa.add_child(feature_group2)
mapa.add_child(feature_group3)
mapa.add_child(feature_group4)
mapa.add_child(feature_group5)
mapa.add_child(folium.map.LayerControl())
folium_static(mapa)

bands = {}
bands[3] = {'LV3':[4,10], 'LV10':[21,56], 'LTRC':[0.15, 2.0], 'LLTX':[0.7, 0.4], 'LLRD':[10, 20], 'LRRD':[10, 20]}
bands[4] = {'LV3':[5,13], 'LV10':[27,71], 'LTRC':[0.15, 2.0], 'LLTX':[0.6, 0.3], 'LLRD':[10, 20], 'LRRD':[10, 20]}
bands[5] = {'LV3':[7,17], 'LV10':[35,93], 'LTRC':[0.15, 2.0], 'LLTX':[0.6, 0.3], 'LLRD':[10, 20], 'LRRD':[10, 20]}
bands[6] = {'LV3':[8,20], 'LV10':[41,110], 'LTRC':[0.15, 2.0], 'LLTX':[0.6, 0.3], 'LLRD':[10, 20], 'LRRD':[10, 20]}

import matplotlib.pyplot as plt
def plotsir(add_text, description):
  f, ax = plt.subplots(1,1,figsize=(12,4))
  #df3.plot(kind='line',x='cumlength',y='LV3',ax=ax)
  if smoothing:
   ax.plot(df3['cumlength'], df3[add_text].rolling(smoothing).mean(), color='b', label='Left lane')
   ax.plot(df4['cumlength'], df4[add_text].rolling(smoothing).mean(), color='r', label='Right lane')
  else:
   ax.plot(df3['cumlength'], df3[add_text], color='b', label='Left lane')
   ax.plot(df4['cumlength'], df4[add_text], color='r', label='Right lane')      
  #ax.plot(t, I, 'y', label='Right lane')
  #ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Radius')

  if hier in bands:  
      if add_text in bands[hier]:
        plt.axhline(y=bands[hier][add_text][0], color='orange', linestyle='-')
        plt.axhline(y=bands[hier][add_text][1], color='r', linestyle='-')

  ax.set_yscale('linear')
  ax.set_xlabel('Chainage  (m) - ' + add_text + ' : ' + description)
  #ax.set_ylabel('%')  # we already handled the x-label with ax1
  #ax2 = ax.twinx()
  color = 'tab:blue'
  #ax2.set_ylabel('Radius (m)', color=color)  # we already handled the x-label with ax1
  #ax2.set_yscale("log")
  #ax2.plot(t, R, alpha=0.4, color=color,label='Radius')
  #ax2.tick_params(axis='y', labelcolor=color)

  #ax.yaxis.set_tick_params(length=0)
  #ax.xaxis.set_tick_params(length=0)
  #ax.grid(b=True, which='major', c='w', lw=2, ls='-')
  legend = ax.legend()
  legend.get_frame().set_alpha(0.5)
  for spine in ('top', 'right', 'bottom', 'left'):
      ax.spines[spine].set_visible(False)
  st.sidebar.pyplot(f)


for param in params_SELECTED:
    #st.write(param.split(' - ')[0])
    plotsir(param.split(' - ')[0], param.split(' - ')[1])
#while True:
#    time.sleep(3)
#    bounds = mapa.get_bounds()
#    df3 = df2[(df2['SECTIONLABEL'] == 'CL1') & (df2['X1'] >= bounds[0][0])  & (df2['X1'] <= bounds[1][0])  & (df2['Y1'] >= bounds[0][1])  & (df2['Y1'] <= bounds[1][1]) ]
#    df4 = df2[(df2['SECTIONLABEL'] == 'CR1') & (df2['X1'] >= bounds[0][0])  & (df2['X1'] <= bounds[1][0])  & (df2['Y1'] >= bounds[0][1])  & (df2['Y1'] <= bounds[1][1]) ]
#    folium_static(mapa)
#    































def RCI(rows, roadclass=None, urban=False):
    #print(len(row.shape))
    if len(rows.shape) > 1:
        row = rows.iloc[0]
    else:
        row = rows
    
    if roadclass is None:
        roadclass = row['LABEL'][0:1]
        if roadclass not in 'ABCU':
            roadclass = 'U'
    
    #Taken from UK Roads, SCANNER surveys for Local Roads, User Guide and Specification, Volume 3
    llrt_or_lrrt_scoring = {'code':['LLRT','LRRT'],'A':[10,20], 'B':[10,20], 'C':[10,20],'U':[10,20],'max':100}
    
    lv3_scoring = {'code':'LV3', 'A':[4,10], 'B':[5,13], 'C':[7,17],'U':[8,20],'max':80}
    
    lv10_scoring = {'code':'LV10', 'A':[21,56], 'B':[27,71], 'C':[35,93],'U':[41,110],'max':60}   

    ltrc_scoring = {'code':'LTRC', 'A':[0.15, 2.0], 'B':[0.15, 2.0], 'C':[0.15, 2.0],'U':[0.15, 2.0],'max':60}   

    if urban:
        if roadclass in ['A','B']:
            lltx_scoring = {'code':'LLTX', 'A':[0.6, 0.3], 'B':[0.6, 0.3], 'C':[0.6,0.3],'U':[0.6,0.3],'max':50}   
        else: #C,U
            lltx_scoring = {'code':'LLTX', 'A':[0.6, 0.3], 'B':[0.6, 0.3], 'C':[0.6,0.3],'U':[0.6,0.3],'max':30}   
    else:
        if roadclass in ['A','B']:
            lltx_scoring = {'code':'LLTX', 'A':[0.7, 0.4], 'B':[0.6, 0.3], 'C':[0.6,0.3],'U':[0.6,0.3],'max':75}   
        else: #C,U
            lltx_scoring = {'code':'LLTX', 'A':[0.7, 0.4], 'B':[0.6, 0.3], 'C':[0.6,0.3],'U':[0.6,0.3],'max':50}   
        
    
