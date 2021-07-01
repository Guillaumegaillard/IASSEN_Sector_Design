#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:50:45 2019

@author: Guillaume Gaillard
"""

import numpy as np
from wrapper_plottings import plottings as guiguiplots
import json
from multiprocessing import Pool,cpu_count
from PyPDF2 import PdfFileMerger
import importlib  
CBparam = importlib.import_module("talon-sector-patterns.CB_params_patterns")

import os
import sys
import importlib  
### locate TalonPy and avoid issues with relative imports there and dashes in dir name 
sys.path.insert(0, './Adaptive-Codebook-Optimization/TalonPyCode/')
TalonPy = importlib.import_module("Adaptive-Codebook-Optimization.TalonPyCode.TalonPy")
from TalonPy import SectorCodebook, MethodIndependent, MethodIndependent_low

#### Some colors
palet_colors=["blue","orange","red","green","blueviolet","brown","burlywood","cadetblue","chartreuse","chocolate","coral","cornflowerblue","crimson","cyan","darkblue","darkcyan",'purple']#"pink","beige","bisque","blanchedalmond",,"yellow"

viridis_hex_ori=["#440154","#440256","#450457","#450559","#46075a","#46085c","#460a5d","#460b5e","#470d60","#470e61","#471063","#471164","#471365","#481467","#481668","#481769","#48186a","#481a6c","#481b6d","#481c6e","#481d6f","#481f70","#482071","#482173","#482374","#482475","#482576","#482677","#482878","#482979","#472a7a","#472c7a","#472d7b","#472e7c","#472f7d","#46307e","#46327e","#46337f","#463480","#453581","#453781","#453882","#443983","#443a83","#443b84","#433d84","#433e85","#423f85","#424086","#424186","#414287","#414487","#404588","#404688","#3f4788","#3f4889","#3e4989","#3e4a89","#3e4c8a","#3d4d8a","#3d4e8a","#3c4f8a","#3c508b","#3b518b","#3b528b","#3a538b","#3a548c","#39558c","#39568c","#38588c","#38598c","#375a8c","#375b8d","#365c8d","#365d8d","#355e8d","#355f8d","#34608d","#34618d","#33628d","#33638d","#32648e","#32658e","#31668e","#31678e","#31688e","#30698e","#306a8e","#2f6b8e","#2f6c8e","#2e6d8e","#2e6e8e","#2e6f8e","#2d708e","#2d718e","#2c718e","#2c728e","#2c738e","#2b748e","#2b758e","#2a768e","#2a778e","#2a788e","#29798e","#297a8e","#297b8e","#287c8e","#287d8e","#277e8e","#277f8e","#27808e","#26818e","#26828e","#26828e","#25838e","#25848e","#25858e","#24868e","#24878e","#23888e","#23898e","#238a8d","#228b8d","#228c8d","#228d8d","#218e8d","#218f8d","#21908d","#21918c","#20928c","#20928c","#20938c","#1f948c","#1f958b","#1f968b","#1f978b","#1f988b","#1f998a","#1f9a8a","#1e9b8a","#1e9c89","#1e9d89","#1f9e89","#1f9f88","#1fa088","#1fa188","#1fa187","#1fa287","#20a386","#20a486","#21a585","#21a685","#22a785","#22a884","#23a983","#24aa83","#25ab82","#25ac82","#26ad81","#27ad81","#28ae80","#29af7f","#2ab07f","#2cb17e","#2db27d","#2eb37c","#2fb47c","#31b57b","#32b67a","#34b679","#35b779","#37b878","#38b977","#3aba76","#3bbb75","#3dbc74","#3fbc73","#40bd72","#42be71","#44bf70","#46c06f","#48c16e","#4ac16d","#4cc26c","#4ec36b","#50c46a","#52c569","#54c568","#56c667","#58c765","#5ac864","#5cc863","#5ec962","#60ca60","#63cb5f","#65cb5e","#67cc5c","#69cd5b","#6ccd5a","#6ece58","#70cf57","#73d056","#75d054","#77d153","#7ad151","#7cd250","#7fd34e","#81d34d","#84d44b","#86d549","#89d548","#8bd646","#8ed645","#90d743","#93d741","#95d840","#98d83e","#9bd93c","#9dd93b","#a0da39","#a2da37","#a5db36","#a8db34","#aadc32","#addc30","#b0dd2f","#b2dd2d","#b5de2b","#b8de29","#bade28","#bddf26","#c0df25","#c2df23","#c5e021","#c8e020","#cae11f","#cde11d","#d0e11c","#d2e21b","#d5e21a","#d8e219","#dae319","#dde318","#dfe318","#e2e418","#e5e419","#e7e419","#eae51a","#ece51b","#efe51c","#f1e51d","#f4e61e","#f6e620","#f8e621","#fbe723","#fde725"]
#https://github.com/stefanv/scale-color-perceptual/blob/master/hex/viridis.json

viridis_hex=[]
indcol=0
for colex in viridis_hex_ori:
    if indcol%16==0:
        viridis_hex.append(colex)
    indcol+=1


#### Init globals, see __main__
filled_boxes=True
colored_boxes=False
plot_types=["2Dmax","2Dmin"]#,"3Dmw","3DmwSCA","3DmwZoom","3DmwZoomSCA"]:
global_rmin=25#15#25#15#7#45#15
global_rmax=45#35#45#37#35#23#75#35
to_pdf_file=True
beam_patterns={}
Global_Sector_Dirs={}
Global_Sector_Names={}
Global_Annotations={}
list_sector_ids=[]
from_json_file=True


#### plot the pattern for a given sector
#### different types of plot are available, see __main__
def plot_pattern(sector_id):

    SectorDirs=Global_Sector_Dirs[sector_id]

    beam_data2D={}
    beam_data3D={}

    if to_pdf_file:
        pp3 = guiguiplots.PdfPages(idv_pdf_name+str(sector_id)+'.pdf')
    
    shortfilename=files[beam_pattern_id] if from_json_file else "CB"
    if len(shortfilename)>20:
        shortfilename=shortfilename[10:]

    # for plot_type in ["2Dmax","2Dmin","3Dmw","3DmwSCA","3DmwZoom","3DmwZoomSCA"]:
    for plot_type in plot_types:
        if plot_type in ["2Dmax","2Dmin"]:
            lbd=len(beam_data2D)    
            beam_data2D[lbd]={
                "values":{}, 
                "axes_projection":"polar",
                "x_axis_label":Global_Sector_Names[sector_id]+' - '+('Azimuths' if plot_type == "2Dmax" else 'Elevations'),
                "colors":palet_colors,
                "theta_min":0.0,
                "theta_max":2*np.pi,
                "rmin":global_rmin,
                "rmax":global_rmax,
                "x_ticks":{
                    "major":{"params":{"bottom":'off',"top":'off',"left":'off',"right":'off'}},
                    "minor":{"params":{"bottom":'off',"top":'off',"left":'off',"right":'off'}},},
                "y_ticks":{
                    "major":{"params":{"bottom":'off',"top":'off',"left":'off',"right":'off','length':0},"labels":[]},
                    "minor":{"params":{"bottom":'off',"top":'off',"left":'off',"right":'off'},"labels":[]},},
            }                       

        if plot_type in ["3Dmw","3DmwSCA","3DmwZoom","3DmwZoomSCA"]:
            lbd=len(beam_data3D)    
            beam_data3D[lbd]={
                "values":{}, 
                "axes_projection":"mollweide",
                "x_axis_label":Global_Sector_Names[sector_id],
                "colors":palet_colors,
                "x_ticks":{
                    "major":{"params":{"bottom":'off',"top":'off',"left":'off',"right":'off'}},
                    "minor":{"params":{"bottom":'off',"top":'off',"left":'off',"right":'off'}},},
                "y_ticks":{
                    "major":{"params":{"bottom":'off',"top":'off',"left":'off',"right":'off'}},
                    "minor":{"params":{"bottom":'off',"top":'off',"left":'off',"right":'off'}},},
            }

        if plot_type in ["2Dmax","2Dmin"]:

            if not from_json_file:
                az,el=np.unravel_index(np.argmax(SectorDirs),SectorDirs.shape)
            else:
                maxdir=0
                for az in SectorDirs:
                    maxdir=max(maxdir, max(SectorDirs[az][el] for el in SectorDirs[az]))

                maxsum=-10000        
                for az in SectorDirs:
                    sumen=np.sum([10**(SectorDirs[az][otherel]/10) for otherel in SectorDirs[az]])
                    if sumen>1:#SectorDirs[az][el]!=-20:
                        maxsum=max(maxsum, max((sumen/(10**(.1*SectorDirs[az][el]))) for el in SectorDirs[az]))

                if plot_type in ["2Dmax","2Dmin"]:
                    broken=False
                    for az in SectorDirs:
                        for el in SectorDirs[az]:
                            if SectorDirs[az][el]==maxdir:
                                broken=True
                                break
                        if broken:
                            break

            #going to boxplot '95%'
            if plot_type in ["2Dmin"]:
                if not from_json_file:
                    if show_highs:
                        beam_data2D[lbd]["values"][len(beam_data2D[lbd]["values"])]={
                            "type":"plot",
                            "y_values":[global_rmin]+[max(SectorDirs[az][zel],global_rmin) for zel in range(181)], 
                            "x_values":[0]+[((x+270)%360)/360*2*np.pi for x in range(181)],
                            'color_index':sector_id%len(beam_data2D[lbd]["colors"]),#int((len(beam_data[lbd]["values"])-2)/3),
                            'linewidth':1.,'ls':"--" } 
                        beam_data2D[lbd]["values"][len(beam_data2D[lbd]["values"])]={
                            "type":"scatter",
                            "y_values":[max(SectorDirs[az][zel],global_rmin) for zel in range(181)], 
                            "x_values":[((x+270)%360)/360*2*np.pi for x in range(181)],
                            'color_index':sector_id%len(beam_data2D[lbd]["colors"]),#int((len(beam_data[lbd]["values"])-2)/3), 
                                                                         's':0.7,'ls':"-", 'marker':'o' } 
                else:
                    beam_data2D[lbd]["values"][len(beam_data2D[lbd]["values"])]={
                        "type":"plot",
                        "y_values":[global_rmin]+[max(SectorDirs[az][str(zel)],global_rmin) for zel in range(181)], 
                        "x_values":[0]+[((x+270)%360)/360*2*np.pi for x in range(181)],
                        'color_index':sector_id%len(beam_data2D[lbd]["colors"]),#int((len(beam_data[lbd]["values"])-2)/3),
                        'linewidth':1.,'ls':"--" } 
                    beam_data2D[lbd]["values"][len(beam_data2D[lbd]["values"])]={
                        "type":"scatter",
                        "y_values":[max(SectorDirs[az][str(zel)],global_rmin) for zel in range(181)], 
                        "x_values":[((x+270)%360)/360*2*np.pi for x in range(181)],
                        'color_index':sector_id%len(beam_data2D[lbd]["colors"]),#int((len(beam_data[lbd]["values"])-2)/3), 
                                                                         's':0.7,'ls':"-", 'marker':'o' } 
                
                for an in Global_Annotations[sector_id]:
                    beam_data2D[lbd]["values"][len(beam_data2D[lbd]["values"])]={
                        "type":"text",
                        "y":45,#an['who'], 
                        "x":((an['elev']+270)%360)/360*2*np.pi,
                        'text':an['who'],
                        "va":'center', "ha":'center',"bbox":dict(boxstyle="round4", fc="white", ec="blue")}


                y_sets=[]
                for zel in range(181):
                    mike=[]
                    bob=[]
                    for azi in range(361):
                        if not from_json_file:
                            if SectorDirs[azi][zel]>global_rmin:
                                mike.append(SectorDirs[azi][zel])
                            bob.append(global_rmin)
                        else:
                            if SectorDirs[str(azi)][str(zel)]>global_rmin:
                                mike.append(SectorDirs[str(azi)][str(zel)])
                            bob.append(global_rmin)
                    if len(mike)>0:
                        y_sets.append(mike)
                    else:
                        y_sets.append(bob)


                if show_boxes:
                    beam_data2D[lbd]["values"][len(beam_data2D[lbd]["values"])]={
                        "type":"boxplot",
                        'x_values':[((x+270)%360)/360*2*np.pi for x in range(181)],
                        'y_sets': y_sets,
                        'whis':(2.5,97.5), 
                        "showfliers":True,#True
                        "showcaps":False,
                        "manage_ticks":False, 
                        'x_size':0.03,                    
                        "linestyle":':',
                        "linewidth":1.2,
                        "fliersize":.7,
                        "box_edge_width":0.5,
                        "fill":filled_boxes,#True,#False,
                        # "fill_color":'cyan',
                        "fill_color":beam_data2D[lbd]["colors"][sector_id%len(beam_data2D[lbd]["colors"])],#'cyan',
                        # "fill_map":(global_rmin,global_rmax),
                        # "cmap_list":viridis_hex,
                        # "color":'cyan'
                        "color":'none',
                        "zorder":3,
                        }
                    if colored_boxes:
                        beam_data2D[lbd]["values"][len(beam_data2D[lbd]["values"])-1]["fill_map"]=(global_rmin,global_rmax)
                    if packman:
                        beam_data2D[lbd]["theta_min"]=-34/180*np.pi
                        beam_data2D[lbd]["theta_max"]=31.75/180*np.pi
                    beam_data2D[lbd]["grid"]={"which":'major',"axis":"both","zorder":0}
                    beam_data2D[lbd]["values"][len(beam_data2D[lbd]["values"])]={
                        "type":"plot",
                        'y_values':[np.mean(y_sets[zel]) for zel in range(181)],
                        'x_values':[((x+270)%360)/360*2*np.pi for x in range(181)],
                        'markersize':1,'marker':'.', 'color':'black',"zorder":3}
                        # 'markersize':1,'marker':'.', 'color_index':sector_id%len(beam_data2D[lbd]["colors"])}
                    # "x_ticks":{"major":{"range_step":1, "from":0, "to":8,"labels":dtype_permut}},   
        

            if plot_type in ["2Dmax"]:
                if not from_json_file:
                    if show_highs:
                        beam_data2D[lbd]["values"][len(beam_data2D[lbd]["values"])]={
                            "type":"plot",
                            "y_values":[global_rmin]+[max(SectorDirs[m][el],global_rmin) for m in range(361)], 
                            "x_values":[0]+[x/360*2*np.pi for x in range(361)],
                            'color_index':sector_id%len(beam_data2D[lbd]["colors"]),#int((len(beam_data[lbd]["values"])-2)/3), 
                            'linewidth':1.,'ls':"--" } 
                        beam_data2D[lbd]["values"][len(beam_data2D[lbd]["values"])]={
                            # "type":"scatter","y_values":[max(SectorDirs[str(m)][el],global_rmin) for m in range(361)], , NS3 or dic dir
                            "type":"scatter","y_values":[max(SectorDirs[m][el],global_rmin) for m in range(361)], 
                            "x_values":[x/360*2*np.pi for x in range(361)],
                            'color_index':sector_id%len(beam_data2D[lbd]["colors"]),#int((len(beam_data[lbd]["values"])-2)/3), 
                            's':0.7,'ls':"-",'marker':'o' } 
                else: #NS3 or dic dir
                    beam_data2D[lbd]["values"][len(beam_data2D[lbd]["values"])]={
                        "type":"plot",
                        "y_values":[global_rmin]+[max(SectorDirs[str(m)][el],global_rmin) for m in range(361)], 
                        "x_values":[0]+[x/360*2*np.pi for x in range(361)],
                        'color_index':sector_id%len(beam_data2D[lbd]["colors"]),#int((len(beam_data[lbd]["values"])-2)/3), 
                        'linewidth':1.,'ls':"--" } 
                    beam_data2D[lbd]["values"][len(beam_data2D[lbd]["values"])]={
                        "type":"scatter","y_values":[max(SectorDirs[str(m)][el],global_rmin) for m in range(361)],
                        "x_values":[x/360*2*np.pi for x in range(361)],
                        'color_index':sector_id%len(beam_data2D[lbd]["colors"]),#int((len(beam_data[lbd]["values"])-2)/3), 
                        's':0.7,'ls':"-",'marker':'o' } 

                for an in Global_Annotations[sector_id]:
                    beam_data2D[lbd]["values"][len(beam_data2D[lbd]["values"])]={
                        "type":"text",
                        "y":45,#an['who'], 
                        "x":an['azim']/360*2*np.pi,
                        'text':an['who'],
                        "va":'center', "ha":'center',"bbox":dict(boxstyle="round4", fc="white", ec="blue")}

                y_sets=[]
                for azi in range(361):
                    mike=[]
                    bob=[]
                    for zel in range(181):
                        if not from_json_file:
                            if SectorDirs[azi][zel]>global_rmin:
                                mike.append(SectorDirs[azi][zel])
                            bob.append(global_rmin)
                        else:
                            if SectorDirs[str(azi)][str(zel)]>global_rmin:
                                mike.append(SectorDirs[str(azi)][str(zel)])
                            bob.append(global_rmin)                            
                    if len(mike)>0:
                        y_sets.append(mike)
                    else:
                        y_sets.append(bob)


                if show_boxes:
                    beam_data2D[lbd]["values"][len(beam_data2D[lbd]["values"])]={
                        "type":"boxplot",
                        'x_values':[x/360*2*np.pi for x in range(361)],
                        'y_sets': y_sets,
                        'whis':(2.5,97.5), 
                        "showfliers":True,
                        "showcaps":False,
                        "manage_ticks":False, 
                        'x_size':0.03,                    
                        "linestyle":':',
                        "linewidth":1.2,
                        "fliersize":.7,
                        "box_edge_width":0.5,
                        "fill":filled_boxes,#False,
                        # "fill_color":'cyan',
                        "fill_color":beam_data2D[lbd]["colors"][sector_id%len(beam_data2D[lbd]["colors"])],#'cyan',
                        # "color":'cyan'
                        "color":'none',#'black',
                        "zorder":3,
                        # "fill_map":(global_rmin,global_rmax),
                        }
                    if colored_boxes:
                        beam_data2D[lbd]["values"][len(beam_data2D[lbd]["values"])-1]["fill_map"]=(global_rmin,global_rmax)
                    if packman:
                        beam_data2D[lbd]["theta_min"]=-160/180*np.pi
                        beam_data2D[lbd]["theta_max"]=160.25/180*np.pi
                    beam_data2D[lbd]["grid"]={"which":'major',"axis":"both","zorder":0}
                    beam_data2D[lbd]["values"][len(beam_data2D[lbd]["values"])]={
                        "type":"plot",
                        'y_values':[np.mean(y_sets[azi]) for azi in range(361)],
                        'x_values':[x/360*2*np.pi for x in range(361)],
                        # 'markersize':1,'marker':'.', 'color_index':sector_id%len(beam_data2D[lbd]["colors"])}
                        'markersize':1,'marker':'.', 'color':'black',"zorder":3}



        if plot_type in ["3DmwZoom","3DmwZoomSCA"]:
            beam_data3D[lbd]["zoom_bbox"]=(-.2/3,.05/3,1/4.1,-1/4.1)
            beam_data3D[lbd]["zoom_bbox_dpi"]=400

        scatter_dots={}
        for colorex in viridis_hex:
            scatter_dots[colorex]={"x":[],"y":[]}

        useless_ys=[]

        for qel in range(181):
            broken=False
            for m in range(361):
                if not from_json_file:
                    if SectorDirs[m][qel]!=-20:
                        broken=True
                        break
                else:
                    if SectorDirs[str(m)][str(qel)]!=-20:
                        broken=True
                        break
            if not broken:
                useless_ys.append(qel)

        useless_xs=[360]


        for m in range(361):
            broken=False
            for qel in range(181):
                if not from_json_file:
                    if SectorDirs[m][qel]!=-20:
                        broken=True
                        break
                else:
                    if SectorDirs[str(m)][str(qel)]!=-20:
                        if m==0:
                            pass
                            # print(qel,(qel-90)/180*np.pi, SectorDirs[str(m)][str(qel)])
                            # SectorDirs[str(m)][str(qel)]=40
                        broken=True
                        break
            if not broken:
                useless_xs.append(m)     

        broken=False
        for m in range(361):
            if m not in useless_xs:
                broken=True
                # print(m,"WAAAA")
                break
        if not broken:
            break

        min_x=1000
        max_x=-1000
        min_y=1000
        max_y=-1000


        if not from_json_file:
            rowsrows=range(len(SectorDirs))
        else:
            rowsrows=SectorDirs

        for dot_pol_x in rowsrows:        
            if not int(dot_pol_x) in useless_xs:
                if not from_json_file:
                    colcol=range(len(SectorDirs[dot_pol_x]))
                else:
                    colcol=SectorDirs[dot_pol_x]
                for dot_pol_y in colcol:
                    # print(dot_pol_y)
                    if not int(dot_pol_y) in useless_ys:
                        if int(dot_pol_x)>179:
                            zex=int(dot_pol_x)-360
                        else:
                            zex=int(dot_pol_x)
                        zey=int(dot_pol_y)-90

                        min_x=min(min_x,zex)
                        max_x=max(max_x,zex)
                        min_y=min(min_y,zey)
                        max_y=max(max_y,zey)
        # print(min_x,max_x,min_y,max_y)

        SectorDirs_list=[]
        for qel in range(min_y,max_y+2):
            azs=[]
            for m in range(min_x,max_x+2):
                if not from_json_file:
                    azs.append(SectorDirs[m+360 if m<0 else m][qel+90])
                else:
                    azs.append(SectorDirs[str(m+360 if m<0 else m)][str(qel+90)])                        
            SectorDirs_list.append(azs)

        # print(SectorDirs_list)

        if plot_type in ["3DmwSCA","3DmwZoomSCA"]:
            ### SCATTER
            for dot_pol_x in rowsrows:
                if not int(dot_pol_x) in useless_xs:
                    if not from_json_file:
                        colcol=range(len(SectorDirs[dot_pol_x]))
                    else:
                        colcol=SectorDirs[dot_pol_x]
                    for dot_pol_y in colcol:
                    # for dot_pol_y in SectorDirs[dot_pol_x]:
                        if not int(dot_pol_y) in useless_ys:
                            power_val=max(SectorDirs[dot_pol_x][dot_pol_y],global_rmin)
                            index_colo=min(int((power_val-global_rmin)/(global_rmax-global_rmin)*len(viridis_hex)),len(viridis_hex)-1)
                            if int(dot_pol_x)>179:
                                scatter_dots[viridis_hex[index_colo]]["x"].append((float(dot_pol_x)-360)/180*np.pi)
                            else:
                                scatter_dots[viridis_hex[index_colo]]["x"].append((float(dot_pol_x))/180*np.pi)
                            # scatter_dots[viridis_hex[index_colo]]["x"].append((float(dot_pol_x)-180)/180*np.pi)
                            scatter_dots[viridis_hex[index_colo]]["y"].append((float(dot_pol_y)-90)/180*np.pi)
            ### SCATTER
            for colorex in viridis_hex:


                beam_data3D[lbd]["values"][len(beam_data3D[lbd]["values"])]={
                    "type":"scatter",
                    "y_values":scatter_dots[colorex]["y"],#[:60], 
                    "x_values":scatter_dots[colorex]["x"],#[:60], 
                    'color':colorex,#int((len(beam_data[lbd]["values"])-2)/3), 
                    's':1.5,#0.2,
                    'ls':"-"
                }

            beam_data3D[lbd]["color_bar"]={
                "color_bounds":np.linspace(global_rmin,global_rmax,len(viridis_hex)+1,endpoint=True),
                "color_list":viridis_hex,
                "title":"dB",
            }

    

        if plot_type in ["3DmwZoom","3Dmw"]:


            ###COLOR MESH
            # #### im = ax.pcolormesh(Lon,Lat,arr, cmap=plt.cm.jet)
            beam_data3D[lbd]["values"][len(beam_data3D[lbd]["values"])]={
                    "type":"pcolormesh",

                    "array":np.array(SectorDirs_list),#np.random.rand(180, 360),#SectorDirs,
                    "cmap_list":viridis_hex,
                    
                    "xmin":min_x,
                    "xmax":max_x+1,
                    "ymin":min_y,
                    "ymax":max_y+1,
                    "vmin":global_rmin,
                    "vmax":global_rmax,
                    
            }

            beam_data3D[lbd]["color_bar"]={
                "color_bounds":np.linspace(global_rmin,global_rmax,len(viridis_hex)+1,endpoint=True),
                "color_list":viridis_hex,
                "title":"dB",
            }


    #different pdf export settings
    if to_pdf_file:
        if not True:
            guiguiplots.conf.set_fig("A4", "landscape")
            guiguiplots.conf.update()
            prepared_plots=guiguiplots.prepare_plots(beam_data)
            guiguiplots.plot_pages(prepared_plots, nb_plots_hor=1, nb_plots_vert=1, show=False, PDF_to_add=pp3) 
        elif True:
            guiguiplots.conf.set_fig("A4", "landscape")
            guiguiplots.conf.update()
            prepared_plots2D=guiguiplots.prepare_plots(beam_data2D)
            prepared_plots3D=guiguiplots.prepare_plots(beam_data3D)
            guiguiplots.plot_pages(prepared_plots2D, nb_plots_hor=2, nb_plots_vert=1, show=False, PDF_to_add=pp3) 
            guiguiplots.plot_pages(prepared_plots3D, nb_plots_hor=1, nb_plots_vert=1, show=False, PDF_to_add=pp3,user_defined_dpi=300) 
        else:
            guiguiplots.conf.set_fig("A4", "portrait")
            guiguiplots.conf.update()
            prepared_plots=guiguiplots.prepare_plots(beam_data)
            guiguiplots.plot_pages(prepared_plots, nb_plots_hor=2, nb_plots_vert=3, show=False, PDF_to_add=pp3) 

        beam_data2D={}
        beam_data3D={}
        pp3.close()
    else:
        return(beam_data2D,beam_data3D)



#### function to prepare a codebook to be plotted
##### different sources are listed
def get_cb_params():
    ###Manual configuration
    return(
        {
        0:{'idx':0,'sid':0,'psh':[1]*32,'etype':[3]*32,'dtype':[1]*8},
        1:{'idx':1,'sid':1,'psh':[2 if i in [8,20,21] else 3 if i ==9 else 0 for i in range(32)],'etype':[3 if i in [8,9,20,21] else 0 for i in range(32)],'dtype':[1]*8},
        2:{'idx':1,'sid':1,'psh':[3 if i ==10 else 0 for i in range(32)],'etype':[3 if i ==10 else 0 for i in range(32)],'dtype':[1]*8}
        })

    ### Default Sector
    mike=SectorCodebook()
    mike.initialize_default(1)
    return({0:mike.get_params(0)})


    ### Two first sectors of default codebook
    mike=MethodIndependent()
    mike.iterate()
    nsec=2
    res={}
    for s in range(nsec):
        res[s]=mike._codebook.get_params(s)
    # print(nsec)
    return(res)


    ### Say you have dl https://github.com/seemoo-lab/talon-library-measurements
    ### all available sectors there:
    res={}
    secid=0
    dirdirdir="/home/guigui/seemoo/talon-library-measurements-master/csi_measurements/"
    for file in os.listdir(dirdirdir):
        toto=json.load(open(dirdirdir+file,'r'))
        measurement=toto['ref_sector_ap_config']
        existing=False
        for existing_sec in res:
            if measurement['psh']==res[existing_sec]['psh'] and measurement['etype']==res[existing_sec]['etype'] and measurement['dtype']==res[existing_sec]['dtype']:
                existing=True                
        if not existing:
            res[secid]={'idx':secid,'sid':secid,'psh':measurement['psh'],'etype':measurement['etype'],'dtype':measurement['dtype']}
            secid+=1


        measurement=toto['ref_sector_st_config']
        existing=False
        for existing_sec in res:
            if measurement['psh']==res[existing_sec]['psh'] and measurement['etype']==res[existing_sec]['etype'] and measurement['dtype']==res[existing_sec]['dtype']:
                existing=True                
        if not existing:
            res[secid]={'idx':secid,'sid':secid,'psh':measurement['psh'],'etype':measurement['etype'],'dtype':measurement['dtype']}
            secid+=1

        # break

    return(res)

if __name__ == '__main__':
    with_ns3=False                                              # Extern source ?
    plotpatterns=True                                           # to plot or not to plot
    show_boxes=True                                             # boxplot based 2D plots
    prepare_new_codebook=True                                   # recompute the steeringvector
    do_single_elements=False                                    # one sector = one ae on
    filled_boxes=True                                           # uniform color on interquartile values
    colored_boxes=False #True                                   # color scale in [2.5% - 97.5%]
    packman=False                                               # backside extrapolation 
    show_highs=False                                            # plot elevation with max of azimuth (and vice-versa)
    plot_types=["2Dmax","2Dmin","3DmwZoom"]                     # includes different representation types
    # plot_types=["2Dmax","2Dmin","3DmwZoom","3Dmw","3DmwSCA","3DmwZoomSCA"]
    # "2Dmax": top view of azimuths planes stacked in 2D
    # "2Dmin": side view of elevations planes stacked in 2D
    # "3Dmw": molleweide view of gains, using pcolormesh
    # "3DmwZoom": molleweide view of gains, using pcolormesh, zoomed to elev -30 +30
    # "3DmwSCA": molleweide view of gains, using scatter plot
    # "3DmwZoomSCA": molleweide view of gains, using scatter plot, zoomed to elev -30 +30


    global_rmin=25#15#25#15#7#45#15                             # min repr gain dB
    global_rmax=45#35#45#37#35#23#75#0835                       # max repr gain dB

    idv_pdf_name='patterns/pattern_param_test'                  # where to export the figs

    to_pdf_file=True                                            # export to a PDF?
    from_json_file=False                                        # use json from talon-pattern


    ## Globals
    beam_patterns={}
    Global_Sector_Dirs={}
    Global_Sector_Names={}
    Global_Annotations={}
    list_sector_ids=[]

    if not with_ns3: # Prepare data in pure Python
        
        if do_single_elements:
            CBparam.prepare_codebook({},single_elements=True,use_ns3=False)

        if prepare_new_codebook:
            zeparams=get_cb_params()
            jsonBPs=CBparam.prepare_codebook(zeparams, use_ns3=False, tofile=from_json_file)
            beam_patterns[len(beam_patterns)]=jsonBPs
            # CBparam.prepare_codebook(zeparams, use_ns3=False, tofile=False)

        if from_json_file:
            beam_patterns={}
            files=["temp_local.json"]
            for file in files:
                beam_patterns[len(beam_patterns)]=json.loads(open(file,"r").read())

        for beam_pattern_id in beam_patterns:
            beam_pattern=beam_patterns[beam_pattern_id]
            
            for antenna in beam_pattern:
                sector_id=0
                for sector in beam_pattern[antenna]["Sectors"]:
                    # if sector_id==-1:#!=8:
                    # if False:#sector_id==0 or (sector_id>32 and sector_id<59):
                    #     sector_id+=1
                    #     continue
                    #     # break
                    Global_Sector_Dirs[sector_id]=beam_pattern[antenna]["Sectors"][sector]["Sector_Directivities"]
                    Global_Sector_Names[sector_id]="Sector "+str(sector_id) #-1 ?
                    Global_Annotations[sector_id]=[]
                    if do_single_elements:
                        Global_Sector_Names[sector_id]="Antenna Element "+str(sector_id+1) #
                    list_sector_ids.append(sector_id)
                    sector_id+=1

                break


    
    else: # Read data prepared elsewhere 

        if prepare_new_codebook:
            # CBparam.prepare_codebook({},single_elements=True)
            CBparam.prepare_codebook(get_cb_params())

        else: ### use a JSON file with NS3 format

            # files=["ns3-802.11ad/codebook_ns3_AP_Param.json"]
            files=["talon-sector-patterns/temp_local.json"]

            for file in files:
                beam_patterns[len(beam_patterns)]=json.loads(open(file,"r").read())

            for beam_pattern_id in beam_patterns:
                beam_pattern=beam_patterns[beam_pattern_id]
                
                for antenna in beam_pattern:

                    sector_id=0
                    for sector in beam_pattern[antenna]["Sectors"]:
                        # if sector_id==-1:#!=8:
                        # if sector_id==0 or (sector_id>32 and sector_id<59):
                        #     sector_id+=1
                        #     continue
                        #     # break
                        Global_Sector_Dirs[sector_id]=beam_pattern[antenna]["Sectors"][sector]["Sector_Directivities"]
                        Global_Sector_Names[sector_id]="Sector "+str(sector_id) #-1 ?
                        Global_Annotations[sector_id]=[]
                        list_sector_ids.append(sector_id)
                        sector_id+=1

                    break

    # make the final plot files 
    if plotpatterns:
        if not os.path.exists('patterns'):
            os.makedirs('patterns')

        # with Pool(processes=4) as p:    # watch RAM    
        with Pool(processes=cpu_count()-1 or 1) as p:
            p.map(plot_pattern, list_sector_ids)
        
        #### Single process version 
        # for sssss in list_sector_ids:
        #     plot_pattern(sssss)

        pdfs = []#['file1.pdf', 'file2.pdf', 'file3.pdf', 'file4.pdf']

        # for sector_id in range(32):
        for sector_id in list_sector_ids:
            pdfs.append(idv_pdf_name+str(sector_id)+'.pdf')


        ## merge them all into one big
        merger = PdfFileMerger()

        for pdf in pdfs:
            merger.append(pdf)

        merger.write(idv_pdf_name+"_elts.pdf")
        merger.close()






