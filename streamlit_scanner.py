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
import numpy as np
#import altair as alt

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
#     SCANNER condition survey results
#     """
)
#with st.echo():

@st.cache(allow_output_mutation=True, ttl=3600)
def load_data():
    gdf_acc = gpd.read_file('IOM_Accidents/IOM Accident Locations.shp')
    gdf_acc.crs = "EPSG:27700"
    def transform_coords(X1,Y1):
        return transformer.transform(X1, Y1)
    
    gdf_acc.loc[:,'Y1'] = gdf_acc['Long']
    gdf_acc.loc[:,'X1'] = gdf_acc['Lat']
    
    
    #gdf_gullies.head()
    df = pd.read_parquet('scannerdata.parquet')

    from pyproj import Proj, transform


    #def reproj(row):
    #  transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
    #  x2,y2 = transformer.transform(row['X1'],row['Y1'])
    #  
    #  row['X1'] = x2
    #  row['Y1'] = y2

    #df = df.apply(lambda x: reproj(x), axis=1)

    df_scrim = pd.read_parquet('scrim.parquet')
    #df = df.sort_values(['SECTIONLABEL','LABEL','STARTCH'])
    
    #gdf = gpd.GeoDataFrame(
    #df, geometry=gpd.points_from_xy(df.Y1, df.X1))
    #gdf.crs = {'init':'epsg:4326'}   
    
    gdf_towns = gpd.read_file('Douglas Area.json')
    gdf_towns.crs = 'EPSG:3857'   
    
    return [df, df_scrim, gdf_acc, gdf_towns]


df, df_scrim, gdf_acc, gdf_towns = load_data()


roads = list(df['roadcode'].unique())

hier_map = {'':'','Primary':3,'District':4,'Local':5,'Access':6, 'Douglas area':0, 'All Roads':7}
hier_selectbox = st.sidebar.selectbox('Hierarchy:',['','Primary','District','Local','Access', 'Douglas area', 'All Roads'])

hier_select = hier_map[hier_selectbox]

if hier_select == '':
    default = ['A5']
else:
    if hier_select == 0: #Douglas area
        default = ['A1', 'A11', 'A18', 'A2', 'A21', 'A22', 'A23', 'A24', 'A25', 'A33',
       'A35', 'A38', 'A39', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46',
       'A47', 'A5', 'A6', 'A8', 'B27', 'B31', 'B34', 'B48', 'B54', 'B55',
       'B56', 'B57', 'B61', 'B62', 'B63', 'B64', 'B65', 'B66', 'B67',
       'B68', 'B69', 'B71', 'B74', 'B75', 'B76', 'B77', 'B80', 'B82',
       'B99900495', 'B99908487', 'C1', 'C10', 'C1025', 'C1026', 'C1062',
       'C1065', 'C1066', 'C1067', 'C1079', 'C1100', 'C1109', 'C112',
       'C1134', 'C1136', 'C1138', 'C1139', 'C1146', 'C1161', 'C1163',
       'C1167', 'C1168', 'C1170', 'C1171', 'C119', 'C1214', 'C1236',
       'C124', 'C1248', 'C126', 'C1281', 'C1356', 'C1358', 'C1366',
       'C1369', 'C1389', 'C1394', 'C1396', 'C1404', 'C1423', 'C1430',
       'C1443', 'C1445', 'C1451', 'C1452', 'C1453', 'C1455', 'C1467',
       'C1468', 'C1473', 'C1475', 'C1477', 'C1478', 'C1484', 'C1489',
       'C1491', 'C1503', 'C1513', 'C1529', 'C1539', 'C1545', 'C1546',
       'C1547', 'C1549', 'C1553', 'C156', 'C158', 'C1600', 'C1602',
       'C164', 'C174', 'C1789', 'C179', 'C1837', 'C1841', 'C1842', 'C2',
       'C22', 'C431', 'C444', 'C448', 'C453', 'C456', 'C460', 'C463',
       'C465', 'C602', 'C605', 'C606', 'C609', 'C612', 'C613', 'C615',
       'C616', 'C617', 'C618', 'C619', 'C620', 'C621', 'C623', 'C624',
       'C625', 'C628', 'C63', 'C632', 'C634', 'C636', 'C637', 'C638',
       'C639', 'C64', 'C641', 'C643', 'C645', 'C647', 'C649', 'C65',
       'C652', 'C656', 'C657', 'C659', 'C666', 'C668', 'C669', 'C67',
       'C672', 'C675', 'C680', 'C683', 'C684', 'C685', 'C686', 'C688',
       'C69', 'C690', 'C693', 'C694', 'C695', 'C697', 'C698', 'C699',
       'C70', 'C701', 'C702', 'C703', 'C704', 'C706', 'C707', 'C708',
       'C714', 'C715', 'C720', 'C721', 'C722', 'C724', 'C729', 'C730',
       'C732', 'C733', 'C734', 'C737', 'C739', 'C740', 'C742', 'C743',
       'C745', 'C746', 'C747', 'C748', 'C751', 'C756', 'C757', 'C758',
       'C759', 'C761', 'C765', 'C766', 'C768', 'C769', 'C77', 'C771',
       'C774', 'C777', 'C78', 'C785', 'C80', 'C83', 'C87', 'C88', 'C89',
       'C90', 'C92', 'C96', 'C99902261', 'C99905296', 'C99905307',
       'C99909011', 'E105', 'E159', 'X99904173', 'X99905154', 'X99905255',
       'X99905913', 'X99909041', 'X99909042', 'X99909047', 'X99909091',
       'C1056', 'C1059', 'C1068', 'C1101', 'C1108', 'C1143', 'C1144',
       'C1153', 'C1156', 'C1158', 'C1159', 'C1160', 'C1169', 'C1235',
       'C1415', 'C1428', 'C1469', 'C1485', 'C1488', 'C1544', 'C1548',
       'C1554', 'C601', 'C610', 'C630', 'C631', 'C642', 'C644', 'C646',
       'C651', 'C658', 'C66', 'C660', 'C661', 'C662', 'C671', 'C673',
       'C678', 'C68', 'C689', 'C696', 'C705', 'C716', 'C718', 'C719',
       'C723', 'C725', 'C727', 'C728', 'C741', 'C750', 'C753', 'C754',
       'C76', 'C763', 'C764', 'C767', 'C773', 'C775', 'C776', 'C79',
       'C97', 'C99900572', 'E182', 'E213', 'X99905215']
    elif hier_select == 7: #all roads
        default = roads
    else:
        default = list(df[df['Class']==hier_select]['roadcode'].unique())


yy = st.sidebar.multiselect("Road:", roads, default=default)

#np.sqrt
df['SD1'] = (df['LTRC']/df['LTRC'].max())*4 - (df['LSUR']/df['LSUR'].max()) 
df['SD2'] = np.sqrt(df['LTRC']/df['LTRC'].max())
df['SD3'] = (df['LTRC']/df['LTRC'].max())

if not yy:
    show_scrim = st.sidebar.checkbox('SCRIM results?')
    y = 'A5'
    df2 = df[df['roadcode']==y]
    selected_chainage = st.slider('Chainage in m', int(df2['cumlength'].min()), int(df2['cumlength'].max()),  \
                              value=(min(11670, max(0,int(df2['cumlength'].max()-1000))),min(17000, int(df2['cumlength'].max()-50))), step=10)

else:
    y = yy[0]
    df2 = df[df['roadcode']==y]
    if len(yy) == 1:
      show_scrim = st.sidebar.checkbox('SCRIM results?')
      selected_chainage = st.slider('Chainage in m', int(df2['cumlength'].min()), int(df2['cumlength'].max()),  \
                              value=(int(df2['cumlength'].min()), int(df2['cumlength'].max())), step=10)
      st.write('Selected chainage:', selected_chainage)
    else:
      selected_chainage = (0,50000)
    
    



params = df.columns[7:49]
with open('Scanner parameters.txt','r') as f:
    available_params = f.readlines()
    available_params = [x.strip() for x in available_params] 
    available_params.append('SD1 - cracking less surf defects')
    available_params.append('SD2 - cracking sqrt')
    available_params.append('SD3 - cracking')
default_selected = [available_params[3],available_params[4],available_params[13],available_params[15],available_params[24],available_params[35],available_params[39]]
if len(yy) <= 1:
    params_SELECTED = st.sidebar.multiselect('Select parameters', available_params, default=default_selected)#params)
smoothing = st.sidebar.slider('Smoothing',0,20,(0))

map_param = st.selectbox("Parameter for map colours:", available_params + ['SCRIM - SCRIM deficiency'], index=41).split(' - ')[0]

df3 = df2[(df2['SECTIONLABEL'] == 'CL1') & (df2['cumlength'] >= selected_chainage[0]) & (df2['cumlength'] <= selected_chainage[1])]
df4 = df2[(df2['SECTIONLABEL'] == 'CR1') & (df2['cumlength'] >= selected_chainage[0]) & (df2['cumlength'] <= selected_chainage[1])]

df3_scrim = df_scrim[(df_scrim['roadcode'] == y) & (df_scrim['XSP'] == 'CL1') & (df_scrim['cumlength'] >= selected_chainage[0]) & (df_scrim['cumlength'] <= selected_chainage[1])]
df4_scrim = df_scrim[(df_scrim['roadcode'] == y) & (df_scrim['XSP'] == 'CR1') & (df_scrim['cumlength'] >= selected_chainage[0]) & (df_scrim['cumlength'] <= selected_chainage[1])]

if smoothing:
    if map_param == 'SCRIM':
      df3_scrim['smoothedmap'] = df3_scrim['THRESHOLD1'].rolling(smoothing).mean().fillna(0)
      df4_scrim['smoothedmap'] = df4_scrim['THRESHOLD1'].rolling(smoothing).mean().fillna(0)
    else:
      df3['smoothedmap'] = df3[map_param].rolling(smoothing).mean().fillna(0)
      df4['smoothedmap'] = df4[map_param].rolling(smoothing).mean().fillna(0)



import folium
from folium.features import DivIcon

feature_group0 = folium.FeatureGroup(name='Left lane')
feature_group1 = folium.FeatureGroup(name='Right lane')

if df3.shape[0]:
    new_coords = [(df3.X1.min()+df3.X1.max())/2, (df3.Y1.min()+df3.Y1.max())/2]
    hier = df3['Class'].iloc[0]
elif df4.shape[0]:
    new_coords = [(df4.X1.min()+df4.X1.max())/2, (df4.Y1.min()+df4.Y1.max())/2]
    hier = df4['Class'].iloc[0]



transformer27 = Transformer.from_crs("epsg:4326", "epsg:3857")

new_coords = transformer.transform(new_coords[0],new_coords[1])

#new_coords = transformer.transform((coords[0]+coords[2])/2,  (coords[1]+coords[3])/2)
#def transform_coords(X1,Y1):
#    return transformer.transform(X1, Y1)

mapa = folium.Map(location=new_coords, tiles="Cartodb Positron", #tiles='https://manngis.gov.im/LocalViewWeb/ArcGIS/Rest/Services/6e0ea2cc-77ed-4fdd-aa1f-80be2daa7d7e/MapServer/tile/{z}/{y}/{x}',attr="MANNGIS IoM Gov",
                  zoom_start=12, prefer_canvas=True)


# feature_group2 = folium.FeatureGroup(name='Gullies at recommended spacing', show=False)
# def plotDot(point):
#     '''input: series that contains a numeric named latitude and a numeric named longitude
#     this function creates a CircleMarker and adds it to your this_map'''
#     #folium.CircleMarker(location=[point.Y1, point.X1],
#     #                    radius=3,
#     #                    weight=1).add_to(mapa)
#     #folium.Marker([point['X1'], point['Y1']],
#     #      #Make color/style changes here
#     #      icon = folium.simple_marker(color='lightgray', marker_icon='oil'),
#     #      ).add_to(mapa)
#     color_map = {'CL1':'blue','CR1':'green'}
    
#     folium.Circle( [point['X1'], point['Y1']], radius=2
#                      , color=color_map[point['SECTIONLABEL']]
#                      , fill_color='lightgray'
#                      , fill=True
#                      ).add_to(feature_group2)
bands = {}

bands[3] = {'LV3':[4,10], 'LV10':[21,56], 'LLTX':[0.7, 0.4], 'LLRD':[10, 20], 'LRRD':[10, 20], 'RCI':[40, 100]}#, 'LTRC':[0.15, 2.0]}
bands[4] = {'LV3':[5,13], 'LV10':[27,71], 'LLTX':[0.6, 0.3], 'LLRD':[10, 20], 'LRRD':[10, 20], 'RCI':[40, 100]}#, 'LTRC':[0.15, 2.0]}
bands[5] = {'LV3':[7,17], 'LV10':[35,93], 'LLTX':[0.6, 0.3], 'LLRD':[10, 20], 'LRRD':[10, 20], 'RCI':[40, 100]}#, 'LTRC':[0.15, 2.0]}
bands[6] = {'LV3':[8,20], 'LV10':[41,110],'LLTX':[0.6, 0.3], 'LLRD':[10, 20], 'LRRD':[10, 20], 'RCI':[40, 100]}#, 'LTRC':[0.15, 2.0]}


from branca.colormap import LinearColormap

color_scale = {}

if hier in bands:  
   if map_param in bands[hier]:
       lim1 = bands[hier][map_param][0]
       lim2 = bands[hier][map_param][1]
       diff = lim2-lim1
       if diff > 0:
           color_scale = LinearColormap(['#91db9b','yellow','red',], index=[max(0,lim1-diff/2),max(0,lim1-min(lim1/2,diff/2)),lim2-diff/6])
       else: #texture
           color_scale = LinearColormap(['red','yellow','#91db9b',], index=[lim2+diff/6,lim1+diff/2,max(0,lim1-diff/2),])
       
       #st.write('limits are %s , %s , %s. %s %s' %(max(0,lim1-diff/2),lim1+diff/2,lim2+diff/6, lim1, lim2))

   elif map_param == 'SCRIM':
       color_scale = LinearColormap(['#91db9b','yellow','red'], index=[0.005,0.05,0.13])            
   else:
       #df_tmp = df3[map_param].append(df4[map_param])
       if smoothing:
        lim1 = df[df['Class']==hier][[map_param]].rolling(smoothing).mean().fillna(0).describe().iloc[4,0]
        lim2 = df[df['Class']==hier][[map_param]].rolling(smoothing).mean().fillna(0).describe().iloc[[2,1],0].sum()*5/3
       else:
        lim1 = df[df['Class']==hier][[map_param]].describe().iloc[4,0]
        lim2 = df[df['Class']==hier][[map_param]].describe().iloc[[2,1],0].sum()*5/3
       diff = lim2-lim1
       color_scale = LinearColormap(['#91db9b','yellow','red',], index=[lim1,lim2-diff/2,lim2-diff/6])       
       #st.write('limits are %s , %s , %s. %s %s' %( lim1,lim1+diff/6,lim2-diff/6, lim1, lim2))
       



feature_group5 = folium.FeatureGroup(name='Area of interest', show=True)
def plotDot(point,color):
    size = 2
    if smoothing:
        to_plot = 'smoothedmap'
    else:
        if map_param == 'SCRIM':
            to_plot = 'THRESHOLD1'
            if float(point[to_plot]) > 0.09:
                size = 8
        else:
            to_plot = map_param
            

    x2,y2 = transformer27.transform(point['X1'],point['Y1'])    
    folium.Circle( [x2, y2], radius=size
                     , color=color_scale(float(point[to_plot])) #'RCIexTex'
                     #, fill_color='black'
                     , fill=True
                     ).add_to(feature_group5)
       
    
feature_group4 = folium.FeatureGroup(name='Chainages', show=True)
def plotChain(point):
    #iframe = folium.IFrame(text, width=700, height=450)
    #popup = folium.Popup(iframe, max_width=3000)
    
    x2,y2 = transformer27.transform(point['X1'],point['Y1'])    
    folium.Marker( [x2, y2], radius=4
                     , color='black'
                     #, fill_color='#808080'
                     #, fill=True
                     , icon=folium.DivIcon(html=str("<p style='font-family:verdana;color:#444;font-size:10px;'>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;%d</p>" % (point['cumlength'])))#, point['LABEL'], point['STARTCH'])))
                     #, popup=str(point['cumlength'])
                     ).add_to(feature_group4)

feature_group6 = folium.FeatureGroup(name='Accidents - wet/damp (blue)', show=False)
feature_group7 = folium.FeatureGroup(name='Accidents - dry (brown)', show=False)
feature_group8 = folium.FeatureGroup(name='Accidents - other (gray)', show=False)
def plotAcc(point):
    if point['Road_Surfa'] == 'Dry':
        carcolor = 'brown'
        fgroup = feature_group7
    elif point['Road_Surfa'] == 'Wet/Damp':
        carcolor = 'blue'
        fgroup = feature_group6
    else:
        carcolor = 'gray'
        fgroup = feature_group8
        
    if point['Overall_co'] == 'Fatal':
        size = 11
    elif point['Overall_co'] == 'Serious':
        size = 9
    elif point['Overall_co'] == 'Slight':
        size = 7
        
        
    folium.Marker( [point['X1'], point['Y1']], radius=4,
        icon=DivIcon(
        icon_size=(20,20),
        #icon_anchor=(7,20),
        html='<div style="font-size: %spt; font-weight: bold; color : %s">▣</div>' % (size, carcolor),
        )
        #icon=folium.Icon(
        #icon_color=carcolor,
        #icon='glyphicon-certificate',    # icon code from above link
        #),#prefix='fa'),  # note fa prefix flag to use Font Awesome
        ).add_to(fgroup)

#use df.apply(,axis=1) to "iterate" through every row in your dataframe
#df2[df2['gullymarker'] ==1].apply(lambda x: plotDot(x), axis = 1)

yx = yy
if yx == None:
    yx = ['A5']
#for road in yx:
if 1:
    if len(yx) == 1:
        df2.iloc[1::15].apply(lambda x: plotChain(x), axis = 1)
    else:
        if hier_select == '':
            df2 = df[df['roadcode'].isin(yx)]
            
            df3_scrim = df_scrim[(df_scrim['roadcode'].isin(yx)) & (df_scrim['XSP'] == 'CL1') & (df_scrim['cumlength'] >= selected_chainage[0]) & (df_scrim['cumlength'] <= selected_chainage[1])]
            df4_scrim = df_scrim[(df_scrim['roadcode'].isin(yx)) & (df_scrim['XSP'] == 'CR1') & (df_scrim['cumlength'] >= selected_chainage[0]) & (df_scrim['cumlength'] <= selected_chainage[1])]
               
        elif hier_select == 0:
            
            gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.Y1, df.X1))
            
            gdf_towns2 = gdf_towns.to_crs(epsg='4326')
            
            df2 = gpd.sjoin(gdf, gdf_towns2, op='intersects', how='inner')
            
            #df2 = df[(df['roadcode'].isin(yx))]
            
            if map_param == 'SCRIM':
                gdf_scrim = gpd.GeoDataFrame(
                    df_scrim, geometry=gpd.points_from_xy(df_scrim.Y1, df_scrim.X1))
                df2_scrim = gpd.sjoin(gdf_scrim, gdf_towns2, op='intersects', how='inner')
                df3_scrim = df2_scrim[(df_scrim['XSP'] == 'CL1') ]
                df4_scrim = df2_scrim[(df_scrim['XSP'] == 'CR1') ]
                   
        else:
            df2 = df[(df['roadcode'].isin(yx)) & (df['Class'] == hier_select)]
            df3_scrim = df_scrim[(df_scrim['roadcode'].isin(yx)) & (df_scrim['Class'] == hier_select) & (df_scrim['XSP'] == 'CL1') & (df_scrim['cumlength'] >= selected_chainage[0]) & (df_scrim['cumlength'] <= selected_chainage[1])]
            df4_scrim = df_scrim[(df_scrim['roadcode'].isin(yx)) & (df_scrim['Class'] == hier_select) & (df_scrim['XSP'] == 'CR1') & (df_scrim['cumlength'] >= selected_chainage[0]) & (df_scrim['cumlength'] <= selected_chainage[1])]
        
        
        
        
        df3 = df2[(df2['SECTIONLABEL'] == 'CL1') & (df2['cumlength'] >= selected_chainage[0]) & (df2['cumlength'] <= selected_chainage[1])]
        df4 = df2[(df2['SECTIONLABEL'] == 'CR1') & (df2['cumlength'] >= selected_chainage[0]) & (df2['cumlength'] <= selected_chainage[1])]
        

        if smoothing:
            if map_param == 'SCRIM':
              df3_scrim['smoothedmap'] = df3_scrim.groupby(['roadcode'])['THRESHOLD1'].rolling(smoothing).mean().reset_index(0,drop=True).fillna(0)
              df4_scrim['smoothedmap'] = df4_scrim.groupby(['roadcode'])['THRESHOLD1'].rolling(smoothing).mean().reset_index(0,drop=True).fillna(0)
            else:
              df3['smoothedmap'] = df3.groupby(['roadcode'])[map_param].rolling(smoothing).mean().reset_index(0,drop=True).fillna(0)
              df4['smoothedmap'] = df4.groupby(['roadcode'])[map_param].rolling(smoothing).mean().reset_index(0,drop=True).fillna(0)
        
        
    
    

    
    spacing = min(int((df3.shape[0]+df4.shape[0])**(2/3)/200)+1,1)
    
    if map_param == 'SCRIM':
      if smoothing:
         df3_scrim.append(df4_scrim).sort_values(['smoothedmap']).iloc[1::spacing].apply(lambda x: plotDot(x,'blue'), axis = 1)      
      else:
         df3_scrim.append(df4_scrim).sort_values(['THRESHOLD1']).iloc[1::spacing].apply(lambda x: plotDot(x,'blue'), axis = 1)
      #df4_scrim.iloc[1::spacing].apply(lambda x: plotDot(x, 'red'), axis = 1)  
    else:
      df3.iloc[1::spacing].apply(lambda x: plotDot(x,'blue'), axis = 1)
      df4.iloc[1::spacing].apply(lambda x: plotDot(x, 'red'), axis = 1)
  
  
  
#if df3.shape[0] > df4.shape[0]:
#    df3.iloc[1::10].apply(lambda x: plotChain(x), axis = 1)
#else:
#    df4.iloc[1::10].apply(lambda x: plotChain(x), axis = 1)

#gdf_gullies.apply(lambda x: plotGul(x), axis = 1)
gdf_acc.apply(lambda x: plotAcc(x), axis = 1)

#mapa.add_child(feature_group2)
mapa.add_child(feature_group4)
mapa.add_child(feature_group5)
mapa.add_child(feature_group6)
mapa.add_child(feature_group7)
mapa.add_child(feature_group8)
mapa.add_child(folium.map.LayerControl())

from folium.plugins import LocateControl
LocateControl().add_to(mapa)

folium_static(mapa)


import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
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
  
  if add_text in ['LSUR','RCIexTex', 'LV3', 'LV10']:
      if df3[add_text].shape[0]:
          if smoothing:
              max1 = df3[add_text].rolling(smoothing).mean().max()
          else:
              max1 = df3[add_text].max()
      else:
          max1 = 0
      if df4[add_text].shape[0]:
          if smoothing:
              max2 = df4[add_text].rolling(smoothing).mean().max()
          else:
              max2 = df4[add_text].max()
      else:
          max2 = 0
      
      if add_text == 'LSUR':
          max3 = 0.35#3.5/(smoothing+1)
          plt.axhline(y=0.35, color='r', linestyle='-')
      elif add_text == 'RCIexTex':
          max3 = 100
          
          
      if add_text == 'LV3':
         max1 = 0
         max2 = 0
         if smoothing:
             max3 = min(30,df4[add_text].rolling(smoothing).mean().max())
         else:
             max3 = min(30, df4[add_text].max())
             
                      
      if add_text == 'LV10':
         max1 = 0
         max2 = 0
         if smoothing:
             max3 = min(150,df4[add_text].rolling(smoothing).mean().max())
         else:
             max3 = min(150, df4[add_text].max())
          
      ax.set_ylim([0, max(max1, max2, max3)])

  #ax.set_ylabel('%')  # we already handled the x-label with ax1
  #ax2 = ax.twinx()
  color = 'tab:blue'
  intervals = float(100)
  loc = plticker.MultipleLocator(base=intervals)
  ax.xaxis.set_minor_locator(loc)
  ax.grid(which='minor', axis='x', linestyle='-', color='0.8')
  ax.grid(which='major', axis='x', linestyle='-', color='0.7')
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


def plot_scrim_top():
  f, ax = plt.subplots(1,1,figsize=(12,4))
  #df3.plot(kind='line',x='cumlength',y='LV3',ax=ax)
  if smoothing:
   ax.plot(df3_scrim['cumlength'], df3_scrim['PARAMETER_'].rolling(smoothing).mean(), color='b', label='Left lane SCRIM coefficient')
   ax.plot(df4_scrim['cumlength'], df4_scrim['PARAMETER_'].rolling(smoothing).mean(), color='r', label='Right lane SCRIM coefficient')
   ax.plot(df3_scrim['cumlength'], df3_scrim['THRESHOLD_'], color='#abb0ff', label='Left lane threshold')
   ax.plot(df4_scrim['cumlength'], df4_scrim['THRESHOLD_'], color='#ffabc1', label='Right lane threshold')
  else:
   ax.plot(df3_scrim['cumlength'], df3_scrim['PARAMETER_'], color='b', label='Left lane SCRIM coefficient')
   ax.plot(df4_scrim['cumlength'], df4_scrim['PARAMETER_'], color='r', label='Right lane SCRIM coefficient') 
   ax.plot(df3_scrim['cumlength'], df3_scrim['THRESHOLD_'], color='#abb0ff', label='Left lane threshold')
   ax.plot(df4_scrim['cumlength'], df4_scrim['THRESHOLD_'], color='#ffabc1', label='Right lane threshold')      
  #ax.plot(t, I, 'y', label='Right lane')
  #ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Radius')
  
  ax.set_yscale('linear')
  ax.set_xlabel('Chainage  (m) - SCRIM coefficients')
  #ax.set_ylabel('%')  # we already handled the x-label with ax1
  #ax2 = ax.twinx()
  color = 'tab:blue'
  intervals = float(100)
  loc = plticker.MultipleLocator(base=intervals)
  ax.xaxis.set_minor_locator(loc)
  ax.grid(which='minor', axis='x', linestyle='-', color='0.8')
  ax.grid(which='major', axis='x', linestyle='-', color='0.7')
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

def plot_scrim_bottom():
    f, ax = plt.subplots(1,1,figsize=(12,4))
    #df3.plot(kind='line',x='cumlength',y='LV3',ax=ax)
    if smoothing:
     ax.plot(df3_scrim['cumlength'], -df3_scrim['THRESHOLD1'].rolling(smoothing).mean(), color='b', label='Left lane SCRIM deficiency')
     ax.plot(df4_scrim['cumlength'], -df4_scrim['THRESHOLD1'].rolling(smoothing).mean(), color='r', label='Right lane SCRIM deficiency')      
    else:
     ax.plot(df3_scrim['cumlength'], -df3_scrim['THRESHOLD1'], color='b', label='Left lane SCRIM deficiency')
     ax.plot(df4_scrim['cumlength'], -df4_scrim['THRESHOLD1'], color='r', label='Right lane SCRIM deficiency')      
  
  
    plt.axhline(y=0.0, color='#dbdbdb', linestyle='-')
    plt.axhline(y=-0.1, color='#8a8a8a', linestyle='-')
    plt.axhline(y=-0.2, color='#303030', linestyle='-')
    
    ax.set_yscale('linear')
    ax.set_xlabel('Chainage  (m) - SCRIM deficiencies')
    #ax.set_ylabel('%')  # we already handled the x-label with ax1
    #ax2 = ax.twinx()
    color = 'tab:blue'
    
    intervals = float(100)
    loc = plticker.MultipleLocator(base=intervals)
    ax.xaxis.set_minor_locator(loc)
    ax.grid(which='minor', axis='x', linestyle='-', color='0.8')
    ax.grid(which='major', axis='x', linestyle='-', color='0.7')
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



if len(yy) <= 1:
    if show_scrim:
      plot_scrim_top()
      plot_scrim_bottom()
    
    st.write(y + ' - ' + df2['Address 1'].mode()[0] )
    for param in params_SELECTED:
        
        plotsir(param.split(' - ')[0], param.split(' - ')[1])
        
        
if st.button('save all'):
    df['smoothed'] = df.groupby(['roadcode','SECTIONLABEL'])[map_param].transform(lambda x: x.rolling(smoothing).mean().fillna(0))
    #df4['smoothedmap'] = df4[map_param].rolling(smoothing).mean()
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Y1, df.X1))
    gdf.to_file('save_%s_smooth-%s.shp' % (map_param, smoothing))
    #df_scrim
                   
#while True:
#    time.sleep(3)
#    bounds = mapa.get_bounds()
#    df3 = df2[(df2['SECTIONLABEL'] == 'CL1') & (df2['X1'] >= bounds[0][0])  & (df2['X1'] <= bounds[1][0])  & (df2['Y1'] >= bounds[0][1])  & (df2['Y1'] <= bounds[1][1]) ]
#    df4 = df2[(df2['SECTIONLABEL'] == 'CR1') & (df2['X1'] >= bounds[0][0])  & (df2['X1'] <= bounds[1][0])  & (df2['Y1'] >= bounds[0][1])  & (df2['Y1'] <= bounds[1][1]) ]
#    folium_static(mapa)
