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

    #df = df.sort_values(['SECTIONLABEL','LABEL','STARTCH'])
    
    #gdf = gpd.GeoDataFrame(
    #df, geometry=gpd.points_from_xy(df.Y1, df.X1))
    #gdf.crs = {'init':'epsg:4326'}   
    
    gdf_towns = gpd.read_file('Douglas Area.json')
    gdf_towns.crs = 'EPSG:3857'   
    
    return [df, gdf_acc, gdf_towns]


df, gdf_acc, gdf_towns = load_data()


roads = list(df['roadcode'].unique())

hier_map = {'':'','Primary':3,'District':4,'Local':5,'Access':6, 'Douglas area':0, 'All':7}
hier_selectbox = st.sidebar.selectbox('Hierarchy:',['','Primary','District','Local','Access', 'Douglas area'])

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
    elif hier_select == 7:
        default = list(df['roadcode'].unique())
    else:
        default = list(df[df['Class']==hier_select]['roadcode'].unique())


yy = st.sidebar.multiselect("Road:", roads, default=default)


if not yy:
    y = 'A5'
    df2 = df[df['roadcode']==y]

else:
    y = yy[0]
    df2 = df[(df['roadcode'].isin(yy))]

    
#SD

cracking_vals = st.sidebar.slider('Cracking', df2['LTRC'].min(), df2['LTRC'].max(),  \
                              value=(df2['LTRC'].min(), df2['LTRC'].max()), step=df2['LTRC'].max()/100)   
cracking_neg = st.sidebar.checkbox('Negative?', key='cracking')
if cracking_neg:
    mult = 1.0
else:
    mult = -1.0

crack_component = (df2[(df2['LTRC'] >= float(cracking_vals[0])) & (df2['LTRC'] <= float(cracking_vals[1]))]['LTRC']/df2['LTRC'].max())*mult


df2['LTRCcomponent'] = (df2['LTRC']/df2['LTRC'].max())*mult
df2.loc[df2.LTRC >= cracking_vals[1], 'LTRCcomponent'] = 0
df2.loc[df2.LTRC <= cracking_vals[0], 'LTRCcomponent'] = 0


LSUR_vals = st.sidebar.slider('Surface defects', df2['LSUR'].min(), df2['LSUR'].max(),  \
                              value=(df2['LSUR'].min(), df2['LSUR'].max()), step=df2['LSUR'].max()/100)   
LSUR_neg = st.sidebar.checkbox('Negative?', key='LSUR')
if LSUR_neg:
    mult = 1.0
else:
    mult = -1.0


df2['LSURcomponent'] = (df2['LSUR']/df2['LSUR'].max())*mult
df2.loc[df2.LSUR >= LSUR_vals[1], 'LSURcomponent'] = 0
df2.loc[df2.LSUR <= LSUR_vals[0], 'LSURcomponent'] = 0
#& (df2['LSUR'] <= LSUR_vals[1])

lim1 = df[['RCIexTex']].describe().iloc[4,0]
lim2 = df[['RCIexTex']].describe().iloc[[2,1],0].sum()*5/3
RCIexTex_vals = st.sidebar.slider('Structural condition', df2['RCIexTex'].min(), df2['RCIexTex'].max(),  \
                              value=(df2['RCIexTex'].min(), df2['RCIexTex'].max()), step=df2['RCIexTex'].max()/100)   
RCIexTex_neg = st.sidebar.checkbox('Negative?', key='struc')
if RCIexTex_neg:
    mult = 1.0
else:
    mult = -1.0

df2['RCIexTexcomponent'] = (df2['RCIexTex']/df2['RCIexTex'].max())*mult
df2.loc[df2.RCIexTex >= RCIexTex_vals[1], 'RCIexTexcomponent'] = 0
df2.loc[df2.RCIexTex <= RCIexTex_vals[0], 'RCIexTexcomponent'] = 0



params = df.columns[7:49]
smoothing = st.sidebar.slider('Smoothing',1,20,(10))


df2['NewParam'] = df2['LTRCcomponent'] + df2['LSURcomponent'] + df2['RCIexTexcomponent']
#(cracking_vals * cracking_neg)

map_param = 'NewParam'#st.selectbox("Parameter for map colours:", params, index=41).split(' - ')[0]

df3 = df2[(df2['SECTIONLABEL'] == 'CL1')]
df4 = df2[(df2['SECTIONLABEL'] == 'CR1')]


if smoothing:
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

#new_coords = transformer.transform((coords[0]+coords[2])/2,  (coords[1]+coords[3])/2)
#def transform_coords(X1,Y1):
#    return transformer.transform(X1, Y1)

mapa = folium.Map(location=new_coords, tiles="Cartodb Positron",
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

bands[3] = {'LV3':[4,10], 'LV10':[21,56], 'LTRC':[0.15, 2.0], 'LLTX':[0.7, 0.4], 'LLRD':[10, 20], 'LRRD':[10, 20], 'RCI':[40, 100]}
bands[4] = {'LV3':[5,13], 'LV10':[27,71], 'LTRC':[0.15, 2.0], 'LLTX':[0.6, 0.3], 'LLRD':[10, 20], 'LRRD':[10, 20], 'RCI':[40, 100]}
bands[5] = {'LV3':[7,17], 'LV10':[35,93], 'LTRC':[0.15, 2.0], 'LLTX':[0.6, 0.3], 'LLRD':[10, 20], 'LRRD':[10, 20], 'RCI':[40, 100]}
bands[6] = {'LV3':[8,20], 'LV10':[41,110], 'LTRC':[0.15, 2.0], 'LLTX':[0.6, 0.3], 'LLRD':[10, 20], 'LRRD':[10, 20], 'RCI':[40, 100]}


from branca.colormap import LinearColormap

color_scale = {}

lim1 = df2[[map_param]].rolling(smoothing).mean().fillna(0).describe().iloc[4,0]
lim2 = df2[[map_param]].rolling(smoothing).mean().fillna(0).describe().iloc[[2,1],0].sum()*5/3
diff = lim2-lim1
color_scale = LinearColormap(['#91db9b','yellow','red',], index=[max(0,lim1-diff/2),max(0,lim1-min(lim1/2,diff/2)),lim2-diff/6])

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
            

        
    folium.Circle( [point['X1'], point['Y1']], radius=size
                     , color=color_scale(float(point[to_plot])) #'RCIexTex'
                     #, fill_color='black'
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

    if smoothing:
          df3['smoothedmap'] = df3.groupby(['roadcode'])[map_param].rolling(smoothing).mean().reset_index(0,drop=True).fillna(0)
          df4['smoothedmap'] = df4.groupby(['roadcode'])[map_param].rolling(smoothing).mean().reset_index(0,drop=True).fillna(0)
    
    



    
    spacing = min(int((df3.shape[0]+df4.shape[0])**(1/3)/200)+1,1)

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


