#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:28:04 2019

@author: Guillaume Gaillard
"""

import numpy as np
import itertools
import pickle

from wrapper_plottings import plottings as guiguiplots
import sys
import importlib  

### locate TalonPy and avoid issues with relative imports there and dashes in dir name 
sys.path.insert(0, './Adaptive-Codebook-Optimization/TalonPyCode/')
TalonPy = importlib.import_module("Adaptive-Codebook-Optimization.TalonPyCode.TalonPy")
from TalonPy import MethodIndependent, MethodIndependent_low

import sector_explorer as optim
import param_plot as paramplot

import json
import time
import os
if not os.path.exists('simu'):
    os.makedirs('simu')

import subprocess

rng = np.random.default_rng()

################# Topologies and their ref. side in meters
TOPO_SCENARIO="old_fixed"
TOPO_SCENARIO="2 para links"
TOPO_SCENARIO="Controlled_Mesh"
TOPO_SCENARIO="BounceNet12"
TOPO_SCENARIO="4 corners"

ref_side=10 # FOR BounceNet12 scenario
ref_side=15 # FOR others
#################


################# Simulation settings (repetitions, iterations, replay, plot patterns...)
channel_trials=100                          # statistical evaluations of channel
mean_behavior=True                          # noise, channel gain, and traffic coefs are fixed average
mean_behavior=False
single_instance_of_pb=True                  # break after t=0
                                            # one single optim evaluation in one given (previous) situation (topo, noise, traffic)
single_instance_of_pb=False
redo=True                                   # same noise, traffic, (dont compute optim if sip and t=0)   
redo=False

redo_figs=True                              # same topo, noise, traffic, (dont compute optim if exists)
redo_figs=False

redo_optim=True                             # same topo, noise, traffic, (but redo optim)
redo_optim=False

redo_figs_ultimate=True                     # same all, everything, including channel gain realizations
redo_figs_ultimate=False

plot_device_patterns=True
plot_device_patterns=False                  # do I plot radiation patterns (may be long)?

plot_device_patterns_iters = 1              # If I plot patterns, for the how many first iterations?

NUM_REPES=1 #10                             # Number of repetitions of the same simulation 
NUM_ITERS=1 #2 #100                         # Number of iterations (run of IASSEN)

nb_devices=4 #32#16#8                       # How many devices (should be pair and correspond to topo choice)


date_simu='2021-01-21-10-35-40'             # Identifier for (the data files saved from) a previous simulation
                                            # will be used only if some redo==True, otherwise set to date of exec 
#################

################# Simulation specific constants and parameters
## Relations MCS, SINR, throughput inspired for IEEE802.11ad standard recommendations
MCSs={
    0:{"sens":-78, "thpt":27.5, "SNR":0, "RSS":-60, "SINR":1},
    1:{"sens":-68, "thpt":385.0, "SNR":4, "RSS":-52, "SINR":8},
    2:{"sens":-66, "thpt":770.0, "SNR":6, "RSS":-48, "SINR":12},
    3:{"sens":-65, "thpt":962.5, "SNR":7, "RSS":-46, "SINR":14},
    4:{"sens":-64, "thpt":1155.0, "SNR":8, "RSS":-44, "SINR":16},
    5:{"sens":-62, "thpt":1251.3, "SNR":8.5, "RSS":-43, "SINR":17},
    6:{"sens":-63, "thpt":1540.0, "SNR":9, "RSS":-42, "SINR":18},
    7:{"sens":-62, "thpt":1925.0, "SNR":10, "RSS":-40, "SINR":20},
    8:{"sens":-61, "thpt":2310.0, "SNR":11, "RSS":-38, "SINR":22},
    9:{"sens":-59, "thpt":2502.5, "SNR":13, "RSS":-34, "SINR":26},
    10:{"sens":-55, "thpt":3080.0, "SNR":18, "RSS":-24, "SINR":36},
    11:{"sens":-54, "thpt":3850.0, "SNR":19, "RSS":-22, "SINR":38},
    12:{"sens":-53, "thpt":4620.0, "SNR":20, "RSS":-20, "SINR":40},
}
SINR_BOUNDS_array=np.array([MCSs[mcs]["SINR"] for mcs in MCSs])

######### channel amplitude attenuation and phase shift
ch_amp_atten_max   = 0.2 # 0 # 20%
ch_phase_shift_max = 0.1 # 0 # radian
#########

######### use log-distance channel model or friis channel model 
log_dist=False
log_dist=True

########## Friis parameters
tx_pow=100 # = around24dBm                            # transmission power
lambda_carrier=0.005 # m                              # wavelength
PLE=4.2                                               # path loss exponent
Other_losses=40                                       # factor for other losses
noise_mean_db=-60                                     # noise mean value (dB)
noise_std_dev_db=4                                    # noise standard deviation (dB)
noise_std_dev_db_ESS=.000002                          # noise standard deviation (dB) as param. taken by IASSEN alg.

########## Interference/traffic load
interf_coef_mean=.5                                   # average load of interfering traffic
interf_coef_var=0 #.25                                # its std. dev. among the devices 
interf_coef_var_simu=0# .05                           # traffic std. dev. in simulation (different for each realization)


########## IASSEN Parameters
OPTIM_ITER_T0=60                                      # Number of Simulated Annealing cycles for the first iteration of IASSEN
OPTIM_ITER_Td0=30                                     # Number of Simulated Annealing cycles for the other iterations of IASSEN

OPTIM_PROCESS_POP=7                                   # Number of IASSEN processes each iteration
const_K=8                                             # SA constant K in temperature expression (initial temperature)


######### Log-distance parameters
d0=1
PL0=10*np.log10(lambda_carrier**2/((4*np.pi)**2 * Other_losses))
Xg_sig=3.92
Xg_sig_mean=3.92#


########## Sector level constraints
max_active_antennas=25#32                            # Maximum number of active antennas
min_active_antennas=6#                               # Minimum number of active antennas


phase_shifts=np.array([0,-np.pi/2,np.pi,np.pi/2])
exp_comp=-1j


############################################################# Functions

def compute_channel_gain(distance):
    if log_dist:
        if mean_behavior:
            return(10**(0.1*PL0) * 10**(-0.1*Xg_sig_mean)/(distance**PLE))
        return(10**(0.1*PL0) * 10**(-0.1*rng.lognormal(0, Xg_sig))/(distance**PLE))
    return(lambda_carrier**2/((4*np.pi)**2 * distance**PLE * Other_losses))

def compute_link_signal(tx_beam_gains, rx_beam_gains, distance):
    g_ch=compute_channel_gain(distance)
    return([max(10**-10,tx_pow*tx_beam_gains[i]*g_ch*rx_beam_gains[i]) for i in range(len(tx_beam_gains))])
    
#### computed each realization
def compute_link_SINRS(signals,devices):

    SNRs={}
    SINRs={}
    for dev in signals:
        SNRs[dev]={}
        SINRs[dev]={}
        if mean_behavior:
            noise=10**(0.1*devices[dev]['noise_mean_db'])
        else:
            noise=10**(0.1*rng.normal(devices[dev]['noise_mean_db'],devices[dev]['noise_std_dev_db']))

        for neigh in signals[dev]:
            if noise!=0:
                SNRs[dev][neigh]=[i/noise for i in signals[dev][neigh]]
            else:
                SNRs[dev][neigh]=[i for i in signals[dev][neigh]]
            
            interf=[0,0,0] # Default, ACO, IASSEN
            for interferer in signals:
                if interferer!=neigh and interferer!=dev:
                    if mean_behavior:
                        local_traffic_coef=devices[interferer]["traffic"]
                    else:
                        local_traffic_coef=max(0.0001,rng.normal(devices[interferer]["traffic"],interf_coef_var_simu))

                    for it in range(len(interf)):
                        interf[it]+=(signals[dev][interferer][it]*local_traffic_coef)

            i_n=[i+noise for i in interf]
            
            SINRs[dev][neigh]=[]
            for it in range(len(interf)):
                if i_n[it]!=0:
                    SINRs[dev][neigh].append(signals[dev][neigh][it]/i_n[it])
                else:
                    SINRs[dev][neigh].append(signals[dev][neigh][it])
        
    return(SNRs,SINRs)

### get closest phase shift
def round_phase(phase):
    return(np.pi/2*np.mod(np.round(phase*2/np.pi)+4, 4))    

### get antenna gains according to the given sector configurations
def compute_beam_gains(devices,dev,neigh):
    # TRANSMISSION
    def_s,ACO_s,optim_s=devices[dev]["tx_sectors"][neigh]
    moduli_tx=[]
    for s in [def_s,ACO_s,optim_s]:
        moduli_tx.append(
            abs(
                sum(
                    devices[dev]["amplitudes"][neigh][s["active_Y"]]*
                    np.exp(
                        (
                            np.array(
                                [-np.pi/2*s["phases"][i] for i in s["active_Y"]]
                            )
                            -   devices[dev]["phases"][neigh][s["active_Y"]]
                        )*exp_comp)
                )
            )#**2
        )
    

    # RECEPTION
    def_s,ACO_s,optim_s=devices[dev]["rx_sectors"][neigh]
    moduli_rx=[]
    for s in [def_s,ACO_s,optim_s]:
        moduli_rx.append(
            abs(
                sum(
                    devices[dev]["amplitudes"][neigh][s["active_Y"]]*
                    np.exp(
                        (
                            np.array(
                                [-np.pi/2*s["phases"][i] for i in s["active_Y"]]
                            )
                            -   devices[dev]["phases"][neigh][s["active_Y"]]
                        )*exp_comp)
                )
            )#**2
        )

    return(moduli_tx,moduli_rx)

### read Default codebook
def fetch_default_sectors():
    mike=MethodIndependent()
    mike.iterate()
    res={}
    for s in [a for a in range(1,32)]+[a for a in range(59,64)]:#range(nsec):32 empty, 58 empty, 64 does not exist
        sec=mike._codebook.get_params(s)
        res[s]={"phases":sec['psh'],"active_Y":np.nonzero(sec['etype'])[0]}
    return(res)

# Update global variable
default_sectors=fetch_default_sectors()

### Simulate IEEE 802.11ad: choose for each dev the default sector with best gain to peer. 
def choose_sector_default(amps,phases):
    output_sectors_AD=[]

    for dev in range(nb_devices):#devices:
        def_gain=-1000
        def_sec=-1
        # def_interfs={}
        for sector in default_sectors:
            def_mod=abs(
                sum(
                    amps[dev,peers[dev],default_sectors[sector]["active_Y"]]*
                    np.exp(
                        (
                            np.array(
                                [-np.pi/2*default_sectors[sector]["phases"][i] for i in default_sectors[sector]["active_Y"]]
                            )
                            -   phases[dev,peers[dev],default_sectors[sector]["active_Y"]]
                        )*exp_comp)
                )
            )
            if def_gain<def_mod:
                def_gain=def_mod
                def_sec=sector

        output_sectors_AD.append(default_sectors[def_sec])

    return(output_sectors_AD)

### Design a sector for each dev as ACO would:
### input: current array characteristics 
def choose_sector_ACO(amps,phases):
    output_sectors_ACO=[]

    # Init an ACO instance
    Oh_MG=MethodIndependent_low()
    Oh_MG._min_active_antennas=min_active_antennas # wao, that's new :)
    Oh_MG._max_active_antennas=max_active_antennas # not restrained to 18 since I don't need training codebooks
    
    for dev in range(nb_devices):#devices:
        ae_gains={}
        ae_gains['Y']=np.arange(32) # I fake a complete training
        ae_gains['amplitude']=amps[dev,peers[dev]] # Current simulated characteristic
        ae_gains['phase']=phases[dev,peers[dev]] # Current simulated characteristic
        
        ae_gains=Oh_MG.choose_tx_sector(ae_gains) # wao, that's new func :)            
        # Deep ACO: let's choose the phase shift that "best" cancel the phase characteristic
        ae_gains['ACO_sec']={
            "phases":[int(np.mod(np.round(-ae_gains['phase'][i]*2/np.pi)+4,4)) for i in ae_gains['Y']],
            "active_Y":ae_gains['ACO_active_Y']}
        
        output_sectors_ACO.append(ae_gains['ACO_sec'])
    return(output_sectors_ACO)


### Let's realize the simulation scenario (alea jacta:))
def prepare_data(NUM_REPES,NUM_ITERS,nb_devices):
    
    # main scenario
    Repes=[str(i) for i in range(NUM_REPES)]
    trials={rep:NUM_ITERS for rep in Repes}

    # variations of antenna elements amplitude characteristics w.r.t. ref array factor
    if ch_amp_atten_max==0:
        amps=np.ones((NUM_REPES,NUM_ITERS,nb_devices,nb_devices,32))
    else:
        amps=rng.uniform(low=1-ch_amp_atten_max, high=1, size=(NUM_REPES,NUM_ITERS,nb_devices,nb_devices,32))
    
    # no self amplitude
    for x in range(nb_devices):    
        amps[:,:,x,x,:]=0.

    # variations of antenna elements phase characteristics w.r.t. ref array factor
    if ch_phase_shift_max==0:
        phases=np.zeros((NUM_REPES,NUM_ITERS,nb_devices,nb_devices,32))
    else:
        phases=rng.uniform(low=-ch_phase_shift_max, high=ch_phase_shift_max, size=(NUM_REPES,NUM_ITERS,nb_devices,nb_devices,32))

    # noise char of devices 
    noises=np.clip(10**(0.1*rng.normal(
        noise_mean_db,noise_std_dev_db_ESS, size=(NUM_REPES,NUM_ITERS,nb_devices))),a_min=10**-20, a_max=None)
    
    # traffic char of devices
    traffics=np.clip(rng.normal(interf_coef_mean,interf_coef_var, size=(NUM_REPES,NUM_ITERS,nb_devices)),a_min=10**-20, a_max=1)

    return(Repes,amps,phases,noises,trials,traffics)


### Let's display the topology in a PDF
def prep_plot_topo(devices):

    legend_heights=np.linspace(-90,90,num=10,endpoint=True)
    norm = guiguiplots.mpl_colors.Normalize(vmin=-90, vmax=90)
    m = guiguiplots.plt.cm.ScalarMappable(norm=norm, cmap=guiguiplots.plt.cm.viridis)

    legend_azim=[
        guiguiplots.mpl_lines.Line2D([],[],color=m.to_rgba(0),marker=r'$\rightarrow$',linestyle='',mew=1.6,ms=8,label='azimuth'),
        guiguiplots.mpl_lines.Line2D([],[],color="k",marker=r'${0}$'.format(nb_devices//2-1),linestyle='',mew=0.6,ms=5,label='AP ID'),
        guiguiplots.mpl_lines.Line2D([],[],color="orange",marker=r'${0}$'.format(nb_devices-1),linestyle='',mew=0.6,ms=5,label='STA ID'),
    ]


    topo={
        0:{
            "values":{}, 
            "y_axis_label":"y coordinates",
            "x_axis_label":"x coordinates",
            "title":"Top view of topology",
            "xmin":-1,
            "ymin":-1,
            "xmax":ref_side+1,
            "ymax":ref_side+1,
            "legends":{"italic_legends":True,"legend_linewidth":3., "legend_labels_font_size":6, "manual_legends":legend_azim},
            "side_bar":{ #legend
                "values":{
                    0:{#height
                        "type":"scatter",
                        'x_values':[-.25]*10,
                        'y_values':legend_heights, 
                        "vmin":-90,
                        "vmax":90,
                        "cmap":guiguiplots.plt.cm.viridis,
                        'c':legend_heights, 
                        's':90, "marker":'*'}, 
                    2:{#elevation
                        "type":"boxplot",
                        'x_values':[0],
                        'y_sets': [[-90,-90,90,90]],
                        'whis':(0,100), 
                        "showfliers":False,
                        "showcaps":False,
                        "manage_ticks":False, 
                        'x_size':0.2,                    
                        "fliersize":4.4,
                        "linewidth":0,
                        "box_edge_width":0,
                        "fill_map":(-90,90),
                        "clip_on_box":True,                                
                        "color":'black',
                        },
                    }, 
                "xmin":-.5,
                "xmax":.25,
                "ymin":-91,          
                "ymax":91,
                "y_axis_label":"height, elevation",
                "y_ticks":{
                    "minor":{
                        "range_step":20, "from":-90, "to":91,
                        "labels":['{0:0.1f}m/{1}°'.format(j,i*20-90) for (i,j) in enumerate(np.linspace(0,3,num=10,endpoint=True))],
                    }
                },
            },
        },
    }

    for dev in devices:
        # arrow showing orientation
        topo[0]["values"][dev]={
            "type":"annotate",
            "text":"",
            "pos":(
                devices[dev]["coords"][0]+np.cos(np.deg2rad(devices[dev]["azimuth"])),
                devices[dev]["coords"][1]+np.sin(np.deg2rad(devices[dev]["azimuth"]))
                ),
            'xytext':(devices[dev]["coords"][0],devices[dev]["coords"][1]), 
            'horizontalalignment':'center', 
            "arrowprops":dict(
                arrowstyle="-|>,head_width=.06, head_length=.12",
                lw=1.5,
                ec=m.to_rgba(devices[dev]["elevation"]),

                shrinkA=0,
                shrinkB=0,
                
                ls="solid",
                zorder=10
                ),
            "color":"orange"
        }

        # label
        topo[0]["values"][nb_devices+dev]={
            "type":"annotate",
            "text":str(dev),
            "pos":(
                devices[dev]["coords"][0]-.75*np.cos(np.deg2rad(devices[dev]["azimuth"])),
                devices[dev]["coords"][1]-.75*np.sin(np.deg2rad(devices[dev]["azimuth"]))
                ),#(3.,3.),
            'horizontalalignment':'center', 
            'verticalalignment':'center', 
            "size": 18,
            "color":"k" if dev< nb_devices/2 else "orange",
        }

        # star for dev
        topo[0]["values"][2*nb_devices+dev]={
            "type":"scatter",
            'x_values':[devices[dev]["coords"][0]],
            'y_values':[devices[dev]["coords"][1]],
            "vmin":0,
            "vmax":3,
            "cmap":guiguiplots.plt.cm.viridis,
            'c':[devices[dev]["coords"][2]], 
            's':200, "marker":'*', "zorder":11

        }

    return(topo)


### Angle helper
def compute_deltas(a0,e0,c0,c1):
    x0,y0,z0=c0
    x1,y1,z1=c1

    dist=((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)**(1/2)
    # print("dist",dist)
    hori_dist=((x1-x0)**2+(y1-y0)**2)**(1/2)
    zero_further=x0**2+y0**2 > x1**2+y1**2

    # if x0==x1 and y0==y1:
    if dist==0:
        return(0,0)
    if x0!=x1:
        da=np.pi-np.arctan2((y1-y0),(x0-x1))-a0

    else:
        if y1>y0:
            da=np.pi/2-a0
        else:
            da=3*np.pi/2-a0

    if z0!=z1:
        rel_el=np.arcsin((z1-z0)/dist)
    else:
        rel_el=0

    de=rel_el-e0 # de is bw -pi and pi
    if de<-np.pi/2:
        print("look behind")
        da+=np.pi
        de+=np.pi
    if de>np.pi/2: 
        print("look behind2")
        da+=np.pi
        de-=np.pi

    return(
        (int(np.rad2deg(da))+360)%360,
        int(np.rad2deg(de))
        )


### let's plot radiation patterns with positions of neighs
def plot_patterns(devices, t):

    protos=["AD","ACO","Optim"]
        
    # data 
    only_CBs_data={}
    only_CBs_data_elevs={}

    # update paramplot globals
    paramplot.to_pdf_file=False
    paramplot.from_json_file=False

    paramplot.CBparam.reload_AF_File=False
    paramplot.CBparam.allowed_multi_pool=True

    for dev in devices:
        only_CBs_data[dev]={}
        only_CBs_data_elevs[dev]={}

        ### reformat sectors for plot paterns
        for protocol in [0,1,2]:#range(4):
            zeparams={}
            dirid=0
            for direction in ["tx_sectors","rx_sectors"]:
                zeparams[dirid]={
                    'sid':dirid,
                    'psh':devices[dev][direction][devices[dev]["peer"]][protocol]['phases'],
                    'etype':[1 if i in devices[dev][direction][devices[dev]["peer"]][protocol]['active_Y'] else 0 for i in range(32)],
                    'dtype':[1 if (
                        (4*i in devices[dev][direction][devices[dev]["peer"]][protocol]['active_Y'])
                        or (4*i+1 in devices[dev][direction][devices[dev]["peer"]][protocol]['active_Y'])
                        or (4*i+2 in devices[dev][direction][devices[dev]["peer"]][protocol]['active_Y'])
                        or (4*i+3 in devices[dev][direction][devices[dev]["peer"]][protocol]['active_Y'])
                        ) else 0 for i in range(8)],
                }
                if protocol<2: # ACO and Default have same tx and rx
                    break
                dirid+=1


            paramplot.beam_patterns=paramplot.CBparam.prepare_codebook(zeparams, use_ns3=False, tofile=False)

            ## See param_plot.py

            paramplot.Global_Sector_Dirs={}
            paramplot.Global_Sector_Names={}
            paramplot.Global_Annotations={}

            paramplot.show_boxes=False
            paramplot.show_boxes=True
            paramplot.prepare_new_codebook=True
            paramplot.do_single_elements=False
            paramplot.filled_boxes=True
            paramplot.colored_boxes=False
            paramplot.packman=False
            paramplot.show_highs=False
            paramplot.plot_types=["2Dmax","2Dmin"]

            ### name the patterns
            for beam_pattern_id in paramplot.beam_patterns:
                beam_pattern=paramplot.beam_patterns[beam_pattern_id]

                sector_id=0
                for sector in beam_pattern["Sectors"]:
                    paramplot.Global_Sector_Dirs[sector_id]=beam_pattern["Sectors"][sector]["Sector_Directivities"]
                    paramplot.Global_Annotations[sector_id]=[]
                    if sector_id==0:
                        paramplot.Global_Sector_Names[sector_id]="Iter {2} - Dev {1} - Tx - {0}".format(
                            protos[protocol],dev, t) 
                    else:
                        paramplot.Global_Sector_Names[sector_id]="Iter {2} - Dev {1} - Rx - {0}".format(
                            protos[protocol],dev, t) 

                    paramplot.Global_Sector_Names[sector_id]=""
                    sector_id+=1

            ### prepare the neighbors info
            for neigh in devices:
                if neigh!=dev:
                    an={
                        "elev":devices[dev]['elev'][neigh],
                        "azim":devices[dev]['azim'][neigh],
                        "who":"[{0}]".format(neigh) if neigh == devices[dev]["peer"] else str(neigh),
                    }
                    paramplot.Global_Annotations[0].append(an)
                    if protocol==2:
                        paramplot.Global_Annotations[1].append(an)


            ### prepare the plots
            pat_tz=paramplot.plot_pattern(0)
            pat_rx_tz={}
            if protocol==2:
                pat_rx_tz=paramplot.plot_pattern(1)

            if 0 in pat_tz[0]:
                pat_tz[0][0]["x_axis_label"]=""
                only_CBs_data[dev][len(only_CBs_data[dev])]=pat_tz[0][0]
            if 1 in pat_tz[0]:
                only_CBs_data_elevs[dev][len(only_CBs_data_elevs[dev])]=pat_tz[0][1]

            if protocol==2:
                if 0 in pat_rx_tz[0]:
                    only_CBs_data[dev][len(only_CBs_data[dev])]=pat_rx_tz[0][0]
                    pat_rx_tz[0][0]["x_axis_label"]=""
                if 1 in pat_rx_tz[0]:
                    only_CBs_data_elevs[dev][len(only_CBs_data_elevs[dev])]=pat_rx_tz[0][1]

            del(pat_tz)
            del(pat_rx_tz)

    # sort the beams
    resorted_beams={}
    labels_beams=["Default - beam pattern - Tx/Rx", "ACO - beam pattern - Tx/Rx", "IASSEN - beam pattern - Tx", "IASSEN - beam pattern - Rx"]
    for dev in range(nb_devices):
        resorted_beams[dev]={}
        for j in only_CBs_data[dev]:
            resorted_beams[dev][len(resorted_beams[dev])]=only_CBs_data[dev][j]
            resorted_beams[dev][len(resorted_beams[dev])-1]["x_axis_label"]=labels_beams[j]

    # plot them
    for dev in range(nb_devices):
        only_CBs=guiguiplots.prepare_plots(resorted_beams[dev])            
        guiguiplots.plot_pages(only_CBs, nb_plots_hor=4, nb_plots_vert=1, grid_specs=[], 
            user_defined_tlfs=8,
            user_defined_alfs=10,
            show=False, PDF_to_add=pp1, page_info="Dev {0}".format(dev))


    del(only_CBs_data)
    del(only_CBs_data_elevs)
    # del(resorted_beams)
    # del(only_CBs)
    return(resorted_beams)


# Let's set the topology
def prepare_devices(Repe):

    # reload if existing
    if single_instance_of_pb or redo_figs or redo_optim:
        try:
            devices=pickle.load(open('simu/{0}_{1}_{2}_optim_analysis.dat'.format(ExpeName,Repe,0), 'rb'))
            return(devices)
        except:
            print("couldnt fetch previous devices, retopologize")
            print(0/0) # just stop a second here, you are trying to redo a simu that has not been done yet

    devices={i:{} for i in range(nb_devices)}
    
    if TOPO_SCENARIO=="old_fixed":
        #5,3,4,2
        devices[0]["coords"]=9,0,2
        devices[1]["coords"]=9,5,2
        devices[2]["coords"]=0,1,2
        devices[3]["coords"]=0,4,2
        devices[0]["id"]=5
        devices[1]["id"]=3
        devices[2]["id"]=4
        devices[3]["id"]=2
        devices[0]["peer_id"]=2
        devices[1]["peer_id"]=4
        devices[2]["peer_id"]=3
        devices[3]["peer_id"]=5
        devices[0]["peer"]=3
        devices[1]["peer"]=2
        devices[2]["peer"]=1
        devices[3]["peer"]=0

        #15,13,14,12
        devices[4]["coords"]=3,9,2
        devices[5]["coords"]=3,0,2
        devices[6]["coords"]=6,9,2
        devices[7]["coords"]=6,0,2
        devices[4]["id"]=15
        devices[5]["id"]=13
        devices[6]["id"]=14
        devices[7]["id"]=12
        devices[4]["peer_id"]=12
        devices[5]["peer_id"]=14
        devices[6]["peer_id"]=13
        devices[7]["peer_id"]=15
        devices[4]["peer"]=7
        devices[5]["peer"]=6
        devices[6]["peer"]=5
        devices[7]["peer"]=4    

    if TOPO_SCENARIO=="BounceNet12": # shelf of APs

        dev_per_axis=np.ceil((nb_devices//2)**(1/2))
        mesh_cell_side_size_x=(ref_side)/dev_per_axis
        mesh_cell_side_size_y=(ref_side-2)/dev_per_axis
        cells=rng.permutation((nb_devices//2))

        ap_segment=ref_side/(nb_devices//2)
        ap_spots=rng.permutation(nb_devices//2)

        for dev in devices:
            if dev >= nb_devices//2:
                cell=cells[dev-nb_devices//2]

                xof=cell%dev_per_axis
                yof=cell//dev_per_axis

                devices[dev]["coords"]=(
                    (xof+rng.random()*.5+.25)*mesh_cell_side_size_x,
                    (yof+rng.random()*.5+.25)*mesh_cell_side_size_y+2,
                    (1.25+rng.random()*.5)
                )
                devices[dev]["azimuth"]=rng.integers(0,360)
                devices[dev]["elevation"]=rng.integers(-15,15)
                # devices[dev]["elevation"]=rng.integers(-90,90)

            else: #APs
                ap_spot=ap_spots[dev]

                xof=ap_spot#%dev_per_axis
                yof=0

                devices[dev]["coords"]=(
                    (xof+rng.random()*.4+.3)*ap_segment,#mesh_cell_side_size,
                    0,
                    (1.25+rng.random()*.5)
                )
                devices[dev]["azimuth"]=rng.integers(40,140)
                devices[dev]["elevation"]=rng.integers(-15,15)


    if TOPO_SCENARIO=="Controlled_Mesh":

        dev_per_axis=np.ceil(nb_devices**(1/2))
        mesh_cell_side_size=ref_side/dev_per_axis
        cells=rng.permutation(nb_devices)

        for dev in devices:
            cell=cells[dev]

            xof=cell%dev_per_axis
            yof=cell//dev_per_axis

            devices[dev]["coords"]=(
                (xof+rng.random()*.5+.25)*mesh_cell_side_size,
                (yof+rng.random()*.5+.25)*mesh_cell_side_size,
                (1.25+rng.random()*.5)
            )
            devices[dev]["azimuth"]=rng.integers(0,360)
            devices[dev]["elevation"]=rng.integers(-15,15)


    if TOPO_SCENARIO=="2 para links":
        ############## "2 para links" scenario
        xcoords=[2,1,13,14]
        ycoords=[1,2,14,13]
        
        zcoords=[0,0,3,3]
        elevs=[11,11,-11,-11]

        azims=[45,45,225,225]

        for dev in devices:
            devices[dev]["coords"]=xcoords[dev],ycoords[dev],zcoords[dev]
            devices[dev]["azimuth"]=azims[dev]
            devices[dev]["elevation"]=elevs[dev]

    if TOPO_SCENARIO=="4 corners":
        ############## "4 corners" scenario
        xcoords=[1,1,14,14]
        ycoords=[14,1,14,1]
        zcoords=[0,1,2,3]
        elevs=[10,3,-3,-10]
        azims=[315,45,225,135] 
        
        for dev in devices:
            devices[dev]["coords"]=xcoords[dev],ycoords[dev],zcoords[dev]
            devices[dev]["azimuth"]=azims[dev]
            devices[dev]["elevation"]=elevs[dev]

    ###### init dev parameters
    for dev in devices:
        devices[dev]["distances"]={}
        devices[dev]["beam_gains"]={}
        devices[dev]["SNRs"]={}
        devices[dev]["SINRs"]={}
        devices[dev]["amplitudes"]={}
        devices[dev]["phases"]={}
        devices[dev]["tx_sectors"]={}
        devices[dev]["rx_sectors"]={}
        devices[dev]["azim"]={} 
        devices[dev]["elev"]={}
        devices[dev]["peer"]=peers[dev]
        #old story of default ip address in 192.168.0.1... 
        devices[dev]["id"]=dev+2
        devices[dev]["peer_id"]=peers[dev]+2
        
    return (devices)

#### Final plots functions
###############################################################################################

#### Simple plot preparation of a SNR/SINR CDF plot 
def plot_cdf(SNR_array, SINR_array):
    legends=["Def","ACO","IASSEN"]
    colors=['r','g','purple']
    cdfplot={
        "values":{},                       
        "x_axis_label":"CDF - dB",
        "title":"",
        "legends":{"italic_legends":True,"legend_loc":"best","legend_labels_font_size":8},
        "xmin":-0.5
    }
    
    for i in range(3): # Default, ACO, IASSEN
        SINR_array[i]=np.where(SINR_array[i]>1,SINR_array[i],1) # avoid undef log values
        SNR_array[i]=np.where(SNR_array[i]>1,SNR_array[i],1)
        cdfplot["values"][2*i]={
            "type":"plot",
            'x_values':sorted(10*np.log10(SNR_array[i])),
            'y_values':np.linspace(0., 1., num=len(SNR_array[i])),
            "color":colors[i],"linestyle":"-", 'legend':legends[i]
        }
        cdfplot["values"][2*i+1]={
            "type":"plot",
            'x_values':sorted(10*np.log10(SINR_array[i])),
            'y_values':np.linspace(0., 1., num=len(SINR_array[i])),
            "color":colors[i],"linestyle":"--", 'legend':legends[i]
        }
    
    return(cdfplot)


#### Plot preparation of a sum of SNR/SINR CDF plot 
def plot_cdf_sum_SINR(SINR_arrays, legend_on=False):
    legends=["Default","ACO","IASSEN"]
    colors=['r','g','purple']
    markers=["x","o","^"]
    cdfplot={
        "values":{},
        "x_axis_label":"Sum of SINRS - CDF (dB)",
        "title":"",
        "legends":{"italic_legends":True,"legend_loc":"best","legend_labels_font_size":8},
        "xmin":-0.5
    }
    
    sumsinrs=[0,0,0]    # Default, ACO, IASSEN
    is_c=[0,1,2] # historical, there, I tried another alg, I had e.g. a [0,1,3]
    for i in range(3):  # Default, ACO, IASSEN
        for j in range(nb_devices):
            sumsinrs[i]+=SINR_arrays[j][is_c[i]]

        cdfplot["values"][i]={
            "type":"plot",
            'x_values':sorted(10*np.log10(sumsinrs[i])),
            'y_values':np.linspace(0., 1., num=len(sumsinrs[i])),
            "color":colors[i],
            "linestyle":"-", 
            "linewidth":1.25, 
            "marker":markers[i],
            "fillstyle":'none',
            "markevery":0.2, 
            "markersize":4, 
        }

        if legend_on:
            cdfplot["values"][i]["legend"]=legends[i]
    
    return(cdfplot)

#### Plot preparation of a mean/min/max of SNR/SINR CDF plot 
def plot_cdf_mean_SINR(SINR_arrays, with_interf=True, legend_on=False):

    legends=["Default","ACO","IASSEN"]
    colors=['r','g','purple']
    markers=["x","o","^"]
    cdfplot={
        "values":{},                       
        "x_axis_label":"Mean target SINR - CDF - dB" if with_interf else "SNR to peer - CDF (dB)",
        "title":"",
        "legends":{"italic_legends":True,"legend_loc":"best","legend_labels_font_size":8},
        "xmin":-0.5
    }
    
    mean_sinrs=[[],[],[]]    # Default, ACO, IASSEN
    min_sinrs=[[],[],[]]    # Default, ACO, IASSEN
    max_sinrs=[[],[],[]]    # Default, ACO, IASSEN
    is_c=[0,1,2]
    for i in range(3):  # Default, ACO, IASSEN
        numtris=len(SINR_arrays[0][is_c[i]])
        for nt in range(numtris):
            mean_sinrs[i].append(np.mean([SINR_arrays[j][is_c[i]][nt] for j in range(nb_devices)]))
            if with_interf:
                min_sinrs[i].append(min([SINR_arrays[j][is_c[i]][nt] for j in range(nb_devices)]))
                max_sinrs[i].append(max([SINR_arrays[j][is_c[i]][nt] for j in range(nb_devices)]))


        cdfplot["values"][i]={
            "type":"plot",
            'x_values':sorted(10*np.log10(mean_sinrs[i])),
            'y_values':np.linspace(0., 1., num=numtris),
            "color":colors[i],
            "linestyle":"-",
            "linewidth":1.25, 
            "marker":markers[i],
            "fillstyle":'none',
            "markevery":0.2, 
            "markersize":4, #'legend':legends[i]
        }
        if with_interf:            
            cdfplot["values"][3+i]={
                "type":"plot",
                'x_values':sorted(10*np.log10(min_sinrs[i])),
                'y_values':np.linspace(0., 1., num=numtris),
                "color":colors[i],"linestyle":"", "marker":"$<$", "markersize":.4
                # 'legend':legends[i]
            }
            cdfplot["values"][6+i]={
                "type":"plot",
                'x_values':sorted(10*np.log10(max_sinrs[i])),
                'y_values':np.linspace(0., 1., num=numtris),
                "color":colors[i],"linestyle":"", "marker":"$>$", "markersize":.4
                 # 'legend':legends[i]
            }

        if legend_on:
            cdfplot["values"][i]["legend"]=legends[i]

    
    return(cdfplot)


#### Plot preparation of a boxplot comparison of SNR/SINR 
def plot_box_SINR(SINR_arrays, snr=False, legend_on=False):

    if nb_devices<10:

        legends=["Default","ACO","IASSEN"]
        colors=['r','g','purple']
        horboxplot={
            "values":{},
            "x_axis_label":"SNRs - dB" if snr else "SINRs - dB" ,
            "title":"",
            "legends":{"italic_legends":True,"legend_loc":"best","legend_labels_font_size":8},
            "x_ticks":{"major":{"range_step":1, "from":0, "to":nb_devices,
                                  "labels":[a for a in range(nb_devices)],
                                  "params":{"direction":'out',"bottom":'on',"top":False,"labelbottom":'on'}}}
            # "xmin":-1.5
        }
        
        is_c=[0,1,2]
        for i in range(3):  # Default, ACO, IASSEN
            horboxplot["values"][i]={
                "type":"boxplot",
                'x_values':np.arange(nb_devices)-.25+i*.25,
                'y_sets':[10*np.log10(SINR_arrays[j][is_c[i]]) for j in range(nb_devices)],
                "x_size":.2,
                "manage_ticks":False,
                "whis":(0,100),
                "color":colors[i],"linestyle":"-", #'legend':legends[i]
            }
            if legend_on:
                horboxplot["values"][i]["legend"]=legends[i]
        
        return(horboxplot)

    else: # too many boxes, resort to min mean median max plots instead 
        legends=["Default","ACO","IASSEN"]
        colors=['r','g','purple']
        horboxplot={
            "values":{},                       
            "x_axis_label":"SNRs - dB" if snr else "SINRs - dB" ,
            "title":"",
            "legends":{"italic_legends":True,"legend_loc":"best","legend_labels_font_size":8},
        }
        
        is_c=[0,1,2]
        for i in range(3):  # Default, ACO, IASSEN
        
            horboxplot["values"][i]={
                "type":"plot",
                'x_values':np.arange(nb_devices),
                'y_values':[min(10*np.log10(0.00001+SINR_arrays[j][is_c[i]])) for j in range(nb_devices)],
                "color":colors[i],"linestyle":"--", #'legend':legends[i]
            }
            horboxplot["values"][3+i]={
                "type":"plot",
                'x_values':np.arange(nb_devices),
                'y_values':[np.mean(10*np.log10(0.00001+SINR_arrays[j][is_c[i]])) for j in range(nb_devices)],
                "color":colors[i],"linestyle":"-", 'linewidth':1.5, #'legend':legends[i]
            }
            horboxplot["values"][6+i]={
                "type":"plot",
                'x_values':np.arange(nb_devices),
                'y_values':[np.median(10*np.log10(0.00001+SINR_arrays[j][is_c[i]])) for j in range(nb_devices)],
                "color":colors[i],"linestyle":":", #'legend':legends[i]
            }
            horboxplot["values"][9+i]={
                "type":"plot",
                'x_values':np.arange(nb_devices),
                'y_values':[max(10*np.log10(0.00001+SINR_arrays[j][is_c[i]])) for j in range(nb_devices)],
                "color":colors[i],"linestyle":"--", #'legend':legends[i]
            }
        
        return(horboxplot)


#### Plot preparation of a boxplot comparison of individual throughputs
def plot_box_tpt(SINR_arrays):
    if nb_devices<10:
        legends=["Default","ACO","IASSEN"]
        colors=['r','g','purple']
        horboxplot={
            "values":{},                       
            "x_axis_label":"Throughput (Mbps/dev.)",
            "title":"",
            "legends":{"italic_legends":True,"legend_loc":"best","legend_labels_font_size":8},
            "x_ticks":{"major":{"range_step":1, "from":0, "to":nb_devices,
                                  "labels":[a for a in range(nb_devices)],
                                  "params":{"direction":'out',"bottom":'on',"top":False,"labelbottom":'on'}}}
            # "xmin":-1.5
        }
        is_c=[0,1,2]
        for i in range(3):  # Default, ACO, IASSEN
            horboxplot["values"][i]={
                "type":"boxplot",
                'x_values':np.arange(nb_devices)-.25+i*.25,
                'y_sets':[[MCSs[mcs]["thpt"] if mcs!=-1 else 0 for mcs in np.digitize(10*np.log10(0.00001+SINR_arrays[j][is_c[i]]),SINR_BOUNDS_array)-1] for j in range(nb_devices)],
                "x_size":.2,
                "manage_ticks":False,
                "whis":(0,100),

                "color":colors[i],"linestyle":"-", #'legend':legends[i]
            }
        
        return(horboxplot)

    else: # too many boxes, resort to mean max plots instead 
        legends=["Default","ACO","IASSEN"]
        colors=['r','g','purple']
        horboxplot={
            "values":{},                       
            "x_axis_label":"Troughputs (Mbps/dev.)",
            "title":"",
            "legends":{"italic_legends":True,"legend_loc":"best","legend_labels_font_size":8},
        }
        
        is_c=[0,1,2]
        for i in range(3):  # Default, ACO, IASSEN
            horboxplot["values"][i+3]={
                "type":"plot",
                'x_values':np.arange(nb_devices),
                'y_values':[
                    np.mean([
                        MCSs[mcs]["thpt"] if mcs!=-1 else 0 for mcs in np.digitize(
                            10*np.log10(0.00001+SINR_arrays[j][is_c[i]]),SINR_BOUNDS_array)-1
                    ]) for j in range(nb_devices)],
                "color":colors[i],"linestyle":"-", 'linewidth':1., #'legend':legends[i]
            }

            horboxplot["values"][i+9]={
                "type":"plot",
                'x_values':np.arange(nb_devices),
                'y_values':[
                    max([
                        MCSs[mcs]["thpt"] if mcs!=-1 else 0 for mcs in np.digitize(
                            10*np.log10(0.00001+SINR_arrays[j][is_c[i]]),SINR_BOUNDS_array)-1
                    ]) for j in range(nb_devices)],
                "color":colors[i],"linestyle":"none", "marker":'.', #'legend':legends[i]
            }
        
        return(horboxplot)

#### Plot preparation of a boxplot comparison of beam gains
def plot_box_bg(beam_gains, to_peer=True):

    ####reshape
    bgsets=[]
    for i in range(3):
        bgp=[]
        for j in range(nb_devices):
            bgd=[]
            bgp.append(bgd)
        bgsets.append(bgp)

    for t in beam_gains:
        for dev in beam_gains[t]:
            for neigh in beam_gains[t][dev]:
                if ((to_peer and dev+neigh==nb_devices-1) or ((not to_peer) and (dev+neigh!=nb_devices-1))):
                    bgsets[0][dev].append(10*np.log10(max(1,beam_gains[t][dev][neigh][0][0])))
                    bgsets[1][dev].append(10*np.log10(max(1,beam_gains[t][dev][neigh][0][1])))
                    bgsets[2][dev].append(10*np.log10(max(1,beam_gains[t][dev][neigh][0][2])))
                    bgsets[2][dev].append(10*np.log10(max(1,beam_gains[t][dev][neigh][1][2])))

    if nb_devices<10:
        legends=["Default","ACO","IASSEN"]
        colors=['r','g','purple']
        horboxplot={
            "values":{},                       
            "x_axis_label":"beam_gains {0} - dB".format("to peer" if to_peer else "to interferers"),
            "title":"",
            "legends":{"italic_legends":True,"legend_loc":"best","legend_labels_font_size":8},
            "x_ticks":{"major":{"range_step":1, "from":0, "to":nb_devices,
                                  "labels":[a for a in range(nb_devices)],
                                  "params":{"direction":'out',"bottom":'on',"top":False,"labelbottom":'on'}}}
            # "xmin":-1.5
        }
        
        is_c=[0,1,2]
        for i in range(3):  # Default, ACO, IASSEN
            horboxplot["values"][i]={
                "type":"boxplot",
                'x_values':np.arange(nb_devices)-.25+i*.25,
                'y_sets':[bgsets[i][j] for j in range(nb_devices)],
                "x_size":.2,
                "manage_ticks":False,
                "whis":(0,100),
                "color":colors[i],"linestyle":"-", #'legend':legends[i]
            }
        
        return(horboxplot)
    else: # too many boxes, resort to min mean median max plots instead 
        legends=["Default","ACO","IASSEN"]
        colors=['r','g','purple']
        horboxplot={
            "values":{},                       
            "x_axis_label":"beam_gains {0} - dB".format("to peer" if to_peer else "to interferers"),
            "title":"",
            "legends":{"italic_legends":True,"legend_loc":"best","legend_labels_font_size":8},
        }
        
        is_c=[0,1,2]
        for i in range(3):  # Default, ACO, IASSEN
            horboxplot["values"][i]={
                "type":"plot",
                'x_values':np.arange(nb_devices),
                'y_values':[min(bgsets[i][j]) for j in range(nb_devices)],
                "color":colors[i],"linestyle":"--", #'legend':legends[i]
            }
            horboxplot["values"][i+3]={
                "type":"plot",
                'x_values':np.arange(nb_devices),
                'y_values':[np.mean(bgsets[i][j]) for j in range(nb_devices)],
                "color":colors[i],"linestyle":"-", 'linewidth':1.5, #'legend':legends[i]
            }
            horboxplot["values"][i+6]={
                "type":"plot",
                'x_values':np.arange(nb_devices),
                'y_values':[np.median(bgsets[i][j]) for j in range(nb_devices)],
                "color":colors[i],"linestyle":":", #'legend':legends[i]
            }
            horboxplot["values"][i+9]={
                "type":"plot",
                'x_values':np.arange(nb_devices),
                'y_values':[max(bgsets[i][j]) for j in range(nb_devices)],
                "color":colors[i],"linestyle":"--", #'legend':legends[i]
            }
        
        return(horboxplot)

#### Plot preparation of aggregated individual throughputs CDF plot
def plot_cdf_sum_throughputs(SINR_arrays, legend_on=False):
    legends=["Default","ACO","IASSEN"]
    colors=['r','g','purple']
    markers=["x","o","^"]
    cdfplot={
        "values":{},                       
        "x_axis_label":"Throughput (Mbps) - CDF",
        "title":"",
        "legends":{"italic_legends":True,"legend_loc":"best","legend_labels_font_size":8},
        "xmin":-0.5
    }
    
    packofsinrs=[np.array([]),np.array([]),np.array([])]    # Default, ACO, IASSEN
    is_c=[0,1,2]
    for i in range(3):  # Default, ACO, IASSEN
        for j in range(nb_devices):
            packofsinrs[i]=np.concatenate((packofsinrs[i],SINR_arrays[j][is_c[i]]))

        deduced_mcs_a=np.digitize(10*np.log10(packofsinrs[i]),SINR_BOUNDS_array)-1
        tpt_list=[MCSs[mcs]["thpt"] if mcs!=-1 else 0 for mcs in deduced_mcs_a]

        cdfplot["values"][i]={
            "type":"plot",
            'x_values':sorted(tpt_list),
            'y_values':np.linspace(0., 1., num=len(packofsinrs[i])),
            "color":colors[i],
            "linestyle":"-",
            "linewidth":1.25, 
            "marker":markers[i],
            "fillstyle":'none',
            "markevery":0.2, 
            "markersize":4, #'legend':legends[i]
        }
        if legend_on:
            cdfplot["values"][i]["legend"]=legends[i]
    
    return(cdfplot)

#### Plot preparation of aggregated upload or download throughputs CDF plot
def plot_cdf_total_throughputs(SINR_arrays,dir_load="up", legend_on=False, xmax=-1):

    legends=["Default","ACO","IASSEN"]
    colors=['r','g','purple']
    markers=["x","o","^"]
    cdfplot={
        "values":{},                       
        "x_axis_label":"{0} (Mbps) - CDF".format("Upload" if dir_load=="up" else "Download"),
        "title":"",
        "legends":{"italic_legends":True,"legend_loc":"best","legend_labels_font_size":8},
        "xmin":-0.5
    }
    if xmax!=-1:
        cdfplot["xmax"]=xmax
    
    total_throughputs=[np.array([]),np.array([]),np.array([])]    # Default, ACO, IASSEN
    is_c=[0,1,2]
    for i in range(3):  # Default, ACO, IASSEN
        if dir_load=="up":
            for j in range(nb_devices//2):
                deduced_mcs_a=np.digitize(10*np.log10(SINR_arrays[j][is_c[i]]),SINR_BOUNDS_array)-1
                tpt_list=[MCSs[mcs]["thpt"] if mcs!=-1 else 0 for mcs in deduced_mcs_a]
                if len(total_throughputs[i])==0:
                    total_throughputs[i]=np.concatenate((total_throughputs[i],tpt_list))
                else:
                    total_throughputs[i]+=tpt_list

        else:
            for j in range(nb_devices//2,nb_devices):
                deduced_mcs_a=np.digitize(10*np.log10(SINR_arrays[j][is_c[i]]),SINR_BOUNDS_array)-1
                tpt_list=[MCSs[mcs]["thpt"] if mcs!=-1 else 0 for mcs in deduced_mcs_a]
                if len(total_throughputs[i])==0:
                    total_throughputs[i]=np.concatenate((total_throughputs[i],tpt_list))
                else:
                    total_throughputs[i]+=tpt_list

        cdfplot["values"][i]={
            "type":"plot",
            'x_values':sorted(total_throughputs[i]),
            'y_values':np.linspace(0., 1., num=len(total_throughputs[i])),
            "color":colors[i],
            "linestyle":"-",
            "linewidth":1.25, 
            "marker":markers[i],
            "fillstyle":'none',
            "markevery":0.2, 
            "markersize":4, #'legend':legends[i]
        }
        if legend_on:
            cdfplot["values"][i]["legend"]=legends[i]
    
    return(cdfplot)

#### Plot preparation of min of per dev. individual throughputs CDF plot
def plot_cdf_min_throughputs(SINR_arrays,dir_load="up"):

    legends=["Default","ACO","IASSEN"]
    colors=['r','g','purple']
    cdfplot={
        "values":{},                       
        "x_axis_label":"CDF - Mbps",
        "title":"",
        "legends":{"italic_legends":True,"legend_loc":"best","legend_labels_font_size":8},
        "xmin":-0.5
    }
    
    total_throughputs=[np.array([]),np.array([]),np.array([])]    # Default, ACO, IASSEN
    is_c=[0,1,2]
    for i in range(3):  # Default, ACO, IASSEN
        if dir_load=="up":
            for j in range(nb_devices//2):
                deduced_mcs_a=np.digitize(10*np.log10(SINR_arrays[j][is_c[i]]),SINR_BOUNDS_array)-1
                tpt_list=[MCSs[mcs]["thpt"] if mcs!=-1 else 0 for mcs in deduced_mcs_a]
                if len(total_throughputs[i])==0:
                    total_throughputs[i]=np.concatenate((total_throughputs[i],tpt_list))
                else:
                    total_throughputs[i]=np.minimum(total_throughputs[i],tpt_list)

        else:
            for j in range(nb_devices//2,nb_devices):
                deduced_mcs_a=np.digitize(10*np.log10(SINR_arrays[j][is_c[i]]),SINR_BOUNDS_array)-1
                tpt_list=[MCSs[mcs]["thpt"] if mcs!=-1 else 0 for mcs in deduced_mcs_a]
                if len(total_throughputs[i])==0:
                    total_throughputs[i]=np.concatenate((total_throughputs[i],tpt_list))
                else:
                    total_throughputs[i]=np.minimum(total_throughputs[i],tpt_list)

        cdfplot["values"][i]={
            "type":"plot",
            'x_values':sorted(total_throughputs[i]),
            'y_values':np.linspace(0., 1., num=len(total_throughputs[i])),
            "color":colors[i],"linestyle":"-", #'legend':legends[i]
        }
    
    return(cdfplot)


#### Plot preparation of median of per dev. individual throughputs CDF plot
def plot_cdf_median_throughputs(SINR_arrays,dir_load="up"):
    legends=["Default","ACO","IASSEN"]
    colors=['r','g','purple']
    cdfplot={
        "values":{},                       
        "x_axis_label":"CDF - Mbps",
        "title":"",
        "legends":{"italic_legends":True,"legend_loc":"best","legend_labels_font_size":8},
        "xmin":-0.5
    }

    total_throughputs=[np.array([]),np.array([]),np.array([])]    # Default, ACO, IASSEN
    is_c=[0,1,2]
    for i in range(3):  # Default, ACO, IASSEN
        if dir_load=="up":
            for j in range(nb_devices//2):
                deduced_mcs_a=np.digitize(10*np.log10(SINR_arrays[j][is_c[i]]),SINR_BOUNDS_array)-1
                tpt_list=[MCSs[mcs]["thpt"] if mcs!=-1 else 0 for mcs in deduced_mcs_a]
                # if sum(total_throughputs[i])==0:
                if len(total_throughputs[i])==0:
                    total_throughputs[i]=np.array(tpt_list)
                else:
                    total_throughputs[i]=np.c_[total_throughputs[i],np.array(tpt_list)]

        else:
            for j in range(nb_devices//2,nb_devices):
                deduced_mcs_a=np.digitize(10*np.log10(SINR_arrays[j][is_c[i]]),SINR_BOUNDS_array)-1
                tpt_list=[MCSs[mcs]["thpt"] if mcs!=-1 else 0 for mcs in deduced_mcs_a]
                # if sum(total_throughputs[i])==0:
                if len(total_throughputs[i])==0:
                    total_throughputs[i]=np.array(tpt_list)
                else:
                    total_throughputs[i]=np.c_[total_throughputs[i],np.array(tpt_list)]

        total_throughputs[i]=np.median(total_throughputs[i],axis=1)

        cdfplot["values"][i]={
            "type":"plot",
            'x_values':sorted(total_throughputs[i]),
            'y_values':np.linspace(0., 1., num=len(total_throughputs[i])),
            "color":colors[i],"linestyle":"-",# 'legend':legends[i]
        }
    
    return(cdfplot)

#### Plot preparation of max of per dev. individual throughputs CDF plot
def plot_cdf_max_throughputs(SINR_arrays,dir_load="up"):
    legends=["Default","ACO","IASSEN"]
    colors=['r','g','purple']
    cdfplot={
        "values":{},                       
        "x_axis_label":"CDF - Mbps",
        "title":"",
        "legends":{"italic_legends":True,"legend_loc":"best","legend_labels_font_size":8},
        "xmin":-0.5
    }
    
    total_throughputs=[np.array([]),np.array([]),np.array([])]    # Default, ACO, IASSEN
    is_c=[0,1,2]
    for i in range(3):  # Default, ACO, IASSEN
        if dir_load=="up":
            for j in range(nb_devices//2):
                deduced_mcs_a=np.digitize(10*np.log10(SINR_arrays[j][is_c[i]]),SINR_BOUNDS_array)-1
                tpt_list=[MCSs[mcs]["thpt"] if mcs!=-1 else 0 for mcs in deduced_mcs_a]
                if len(total_throughputs[i])==0:
                    total_throughputs[i]=np.concatenate((total_throughputs[i],tpt_list))
                else:
                    total_throughputs[i]=np.maximum(total_throughputs[i],tpt_list)

        else:
            for j in range(nb_devices//2,nb_devices):
                deduced_mcs_a=np.digitize(10*np.log10(SINR_arrays[j][is_c[i]]),SINR_BOUNDS_array)-1
                tpt_list=[MCSs[mcs]["thpt"] if mcs!=-1 else 0 for mcs in deduced_mcs_a]
                if len(total_throughputs[i])==0:
                    total_throughputs[i]=np.concatenate((total_throughputs[i],tpt_list))
                else:
                    total_throughputs[i]=np.maximum(total_throughputs[i],tpt_list)

        cdfplot["values"][i]={
            "type":"plot",
            'x_values':sorted(total_throughputs[i]),
            'y_values':np.linspace(0., 1., num=len(total_throughputs[i])),
            "color":colors[i],"linestyle":"-", #'legend':legends[i]
        }
    
    return(cdfplot)
###############################################################################################



        
if __name__ == "__main__":

    # Should have been created when paramplot module called CBparams module...
    steering_array=pickle.load(open('steeringVector_as_array.dat','rb'))

    # redo things or do new things?
    if (single_instance_of_pb and redo) or redo_figs or redo_optim:
        # start_time='2020-12-08-14-07-39'
        start_time=date_simu
    else:
        start_time=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    
    ExpeName=start_time+'_scalingUP'

    ### Open PDFs
    optim.pp=optim.guiguiplots.PdfPages("simu/{0}_plots_optim.pdf".format(start_time))
    optim.ExpeName=ExpeName
    pp1 = guiguiplots.PdfPages('simu/{0}_SNR_model.pdf'.format(ExpeName))#.format(ExpeName,Repe))

    # init data
    cdfplots={}
    sumsinr_plot={}
    plotted_beams={}
    beam_gains={}
    objective_funcs=np.zeros((NUM_REPES,NUM_ITERS,3))

    # init multi repe
    aggregated_multi_repe={}
    aggregated_multi_repe["SINRs"]=[]
    aggregated_multi_repe["SNRs"]=[]

    # peers=[7,6,5,4,3,2,1,0]
    peers=list(range(nb_devices))[::-1]

    # get simu instance
    if (single_instance_of_pb and redo) or redo_figs or redo_optim:
        mike = pickle.load(open('simu/{0}_optim_data.dat'.format(ExpeName), 'rb'))
    else:
        mike = prepare_data(NUM_REPES,NUM_ITERS,nb_devices) #, from_angles=True
        pickle.dump(mike, open('simu/{0}_optim_data.dat'.format(ExpeName), 'wb'))
    try:
        Repes,ch_amps,ch_phases,noises,trials,traffics = mike
    except:
        Repes,ch_amps,ch_phases,noises,trials = mike # before January 28th 2021, should not happen again
        traffics=interf_coef_mean*np.ones((NUM_REPES,NUM_ITERS,nb_devices))

    # copy things 
    amps=ch_amps[:]
    phases=ch_phases[:]


    # let's begin
    for repid in range(NUM_REPES):
        Repe=str(repid)
        beam_gains[repid]={}
    
        devices=prepare_devices(Repe)

        topo=prep_plot_topo(devices)
        prepared_topo=guiguiplots.prepare_plots(topo)
        guiguiplots.plot_pages(prepared_topo, nb_plots_hor=1, nb_plots_vert=1, show=False, PDF_to_add=pp1)
        pptemp = guiguiplots.PdfPages('temptopo.pdf')#.format(ExpeName,Repe))
        guiguiplots.plot_pages(prepared_topo, nb_plots_hor=1, nb_plots_vert=1, show=False, PDF_to_add=pptemp)
        pptemp.close()

        # init sinrs
        if not redo_figs_ultimate:
            for dev in devices:
                for neigh in devices:
                    if dev!=neigh:
                        if single_instance_of_pb:
                            devices[dev]["SNRs"][neigh]=[np.zeros(channel_trials),np.zeros(channel_trials),np.zeros(channel_trials),np.zeros(channel_trials)]
                            devices[dev]["SINRs"][neigh]=[np.zeros(channel_trials),np.zeros(channel_trials),np.zeros(channel_trials),np.zeros(channel_trials)]
                        else:
                            devices[dev]["SNRs"][neigh]=[np.zeros(channel_trials*trials[Repe]),np.zeros(channel_trials*trials[Repe]),np.zeros(channel_trials*trials[Repe]),np.zeros(channel_trials*trials[Repe])]
                            devices[dev]["SINRs"][neigh]=[np.zeros(channel_trials*trials[Repe]),np.zeros(channel_trials*trials[Repe]),np.zeros(channel_trials*trials[Repe]),np.zeros(channel_trials*trials[Repe])]                
                        devices[dev]["distances"][neigh]=(
                            (devices[dev]["coords"][0]-devices[neigh]["coords"][0])**2
                            +(devices[dev]["coords"][1]-devices[neigh]["coords"][1])**2
                            +(devices[dev]["coords"][2]-devices[neigh]["coords"][2])**2
                            )**(.5)

                        

        else:
            devices=pickle.load(open('simu/{0}_{1}_{2}_optim_analysis.dat'.format(ExpeName,Repe,trials[Repe]-1), 'rb'))


        # let's loop over iterations (trials)        
        for t in range(trials[Repe]):
            if redo_figs_ultimate: # already all done
                break

            print("Repe,t:",Repe,t)

            ## Dev characteristics
            for dev in devices:
                devices[dev]["noise_mean_db"]=noise_mean_db
                devices[dev]["noise_std_dev_db"]=noise_std_dev_db
            for dev in devices:
                for neigh in devices:
                    if dev!=neigh:

                        ### change format (bounds) of angles to locate neighs in spherical
                        Deltazim,DeltaElev=compute_deltas(
                            np.deg2rad(devices[dev]["azimuth"]),
                            np.deg2rad(devices[dev]["elevation"]),
                            devices[dev]["coords"],
                            devices[neigh]["coords"]
                        )
                        DeltaElev+=90
                        devices[dev]['elev'][neigh]=DeltaElev
                        devices[dev]['azim'][neigh]=Deltazim
    
                        ### instanciate amp and phase char 
                        # from array factor
                        ref_amps_of_neigh_at_dev=(abs(
                            steering_array[Deltazim,DeltaElev,:,0]+steering_array[Deltazim,DeltaElev,:,1]*1j))
                        ref_phases_of_neigh_at_dev=np.angle(
                            steering_array[Deltazim,DeltaElev,:,0]+steering_array[Deltazim,DeltaElev,:,1]*1j)

                        # and simu variations
                        devices[dev]["amplitudes"][neigh]=ch_amps[repid,t,dev,neigh]*ref_amps_of_neigh_at_dev
                        amps[repid,t,dev,neigh]*=ref_amps_of_neigh_at_dev

                        devices[dev]["phases"][neigh]=ref_phases_of_neigh_at_dev+ch_phases[repid,t,dev,neigh]
                        for aep in range(len(devices[dev]["phases"][neigh])):
                            if devices[dev]["phases"][neigh][aep] > np.pi :
                                devices[dev]["phases"][neigh][aep]-=2*np.pi
                            phases[repid,t,dev,neigh,aep]=devices[dev]["phases"][neigh][aep]
            # break

            ######## Dict of params for Optim module
            config_dict={
                "log_dist":log_dist,
                "tx_pow":tx_pow,
                "lambda_carrier":lambda_carrier,
                "PLE":PLE,
                "Other_losses":Other_losses,
                "interf_coef_mean":interf_coef_mean,
                "interf_coefs":traffics[repid,t],

                "d0":d0,
                "PL0":PL0,
                "Xg_sig_mean":Xg_sig_mean,

                "max_active_antennas":max_active_antennas,
                "min_active_antennas":min_active_antennas,

                "phase_shifts":phase_shifts,
                "const_K": const_K,#8,

                "peers":peers,
                "noises":noises[repid,t]
            }

            ########## Design Default and ACO sectors
            csAD=choose_sector_default(amps[repid,t],phases[repid,t])
            csACO=choose_sector_ACO(amps[repid,t],phases[repid,t])
            
            ########## Design IASSEN sectors
            if t==0: 
                ### set the variables for optim
                optim.update_np(devices, amps[repid,t],phases[repid,t], config_dict)

                ### if we are repeating a previous simu
                if (single_instance_of_pb and redo) or redo_figs:
                    try:                        
                        old_dev_data=pickle.load(open('simu/{0}_{1}_{2}_optim_analysis.dat'.format(ExpeName,Repe,t), 'rb'))
                    except: # in fact, we are not repeating smth yet, first time here
                        old_dev_data={}
                        old_idv=None

                    if len(old_dev_data)>0:
                        old_idv=np.zeros((nb_devices,2,32),dtype='int')
                        # old_idv_unshifted=np.zeros((nb_devices,2,32),dtype='int')

                        ### fetch the previous sectors chosen
                        dicold={"AD":1,"ACO":2,"DIA":-2,"optim":-1}
                        old_wished="optim"
                        cifra=dicold[old_wished]
                        for od in itertools.product(range(nb_devices), range(2)):
                            acty=np.array(old_dev_data[od[0]]['tx_sectors' if od[1]==0 else 'rx_sectors'][devices[od[0]]["peer"]][cifra]['active_Y'])
                            phase_array=np.array(old_dev_data[od[0]]['tx_sectors' if od[1]==0 else 'rx_sectors'][devices[od[0]]["peer"]][cifra]['phases'])
                            if len(acty)>0:
                                old_idv[od][acty]=((phase_array[acty]-phase_array[min(acty)]+4)%4)+1
                            # old_idv_unshifted[od][acty]=phase_array[acty]+1

                        old_idv_secs_shape=old_idv.copy()
                        old_idv = old_idv.reshape(nb_devices*2*32,)
                  

                    # use the previous sector design
                    if redo_figs:
                        optim_sectors=old_idv_secs_shape
                    else: # call to IASSEN
                        optim_sectors=optim.get_optim_sectors(
                            strategy="simu_anneal_with_pre_gibbs", #that's IASSEN
                            maxiter=OPTIM_ITER_T0, 
                            population_size=OPTIM_PROCESS_POP,
                            initial_idv=old_idv)
                else: # call to IASSEN
                    optim_sectors=optim.get_optim_sectors(
                        strategy="simu_anneal_with_pre_gibbs",maxiter=OPTIM_ITER_T0, population_size=OPTIM_PROCESS_POP)
                # print(optim_sectors)
            
            else: #same for t!=0
                # fetch the previous sector design if exists
                if redo_figs: 
                    try:                            
                        old_dev_data=pickle.load(open('simu/{0}_{1}_{2}_optim_analysis.dat'.format(ExpeName,Repe,t), 'rb'))
                    except:
                        old_dev_data={}
                        old_idv=None
                        break

                    if len(old_dev_data)>0:
                        old_idv=np.zeros((nb_devices,2,32),dtype='int')
                        dicold={"AD":1,"ACO":2,"DIA":-2,"optim":-1}
                        old_wished="optim"
                        cifra=dicold[old_wished]
                        for od in itertools.product(range(nb_devices), range(2)):
                            acty=np.array(old_dev_data[od[0]]['tx_sectors' if od[1]==0 else 'rx_sectors'][devices[od[0]]["peer"]][cifra]['active_Y'])
                            phase_array=np.array(old_dev_data[od[0]]['tx_sectors' if od[1]==0 else 'rx_sectors'][devices[od[0]]["peer"]][cifra]['phases'])
                            if len(acty)>0:
                                old_idv[od][acty]=((phase_array[acty]-phase_array[min(acty)]+4)%4)+1                        
                        optim_sectors=old_idv
                else:
                    # topo did not change, but the rest is t-depedent
                    optim.update_amps_and_phases_np(devices, amps[repid,t],phases[repid,t],traffics[repid,t])
                    # IASSEN
                    optim_sectors=optim.get_optim_sectors(
                        strategy="simu_anneal_with_pre_gibbs",
                        maxiter=OPTIM_ITER_Td0, 
                        population_size=OPTIM_PROCESS_POP, 
                        initial_idv=optim_sectors.reshape(nb_devices*2*32,))
            
            ##### RESHAPE THE resulting optim sectors
            reformatted_optim_sectors_rx=[
                {
                 'phases': np.clip(optim_sectors[sec_id][1]-1,a_min=0, a_max=None), 
                 'active_Y': np.nonzero(optim_sectors[sec_id][1])[0]
                }
                for sec_id in range(nb_devices)]

            reformatted_optim_sectors_tx=[
                {
                 'phases': np.clip(optim_sectors[sec_id][0]-1,a_min=0, a_max=None), 
                 'active_Y': np.nonzero(optim_sectors[sec_id][0])[0]
                }
                for sec_id in range(nb_devices)]


            #### Reshape Default sectors
            csad_array=np.zeros((nb_devices,2,32),dtype='int')
            for od in itertools.product(range(nb_devices), range(2)):
                acty=csAD[od[0]]["active_Y"]
                phase_array=np.array(csAD[od[0]]["phases"])
                if len(acty)>0:
                    csad_array[od][acty]=phase_array[acty]+1

            #### Reshape ACO's sectors
            csaco_array=np.zeros((nb_devices,2,32),dtype='int')
            for od in itertools.product(range(nb_devices), range(2)):
                acty=csACO[od[0]]["active_Y"]
                phase_array=np.array(csACO[od[0]]["phases"])
                if len(acty)>0:
                    csaco_array[od][acty]=phase_array[acty]+1

            ### Store results 
            objective_funcs[repid,t,:]=[optim.sum_SINRS(csad_array),optim.sum_SINRS(csaco_array),optim.sum_SINRS(optim_sectors)]

            ### update peer links
            for dev in devices:#[0]:#
                for neigh in devices:#[1]:#
                    if dev!=neigh:
                        if neigh==devices[dev]["peer"]:
                            devices[dev]["tx_sectors"][neigh]=csAD[dev],csACO[dev],reformatted_optim_sectors_tx[dev]
                            devices[dev]["rx_sectors"][neigh]=csAD[dev],csACO[dev],reformatted_optim_sectors_rx[dev]

            ### update interference links
            for dev in devices:#[0]:#
                for neigh in devices:#[1]:#
                    if dev!=neigh:
                        if neigh!=devices[dev]["peer"]:
                            devices[dev]["tx_sectors"][neigh]=devices[dev]["tx_sectors"][devices[dev]["peer"]]
                            devices[dev]["rx_sectors"][neigh]=devices[dev]["rx_sectors"][devices[dev]["peer"]]
                            
            beam_gains[repid][t]={}
            
            ### compute beam gains for all links        
            for dev in devices:
                beam_gains[repid][t][dev]={}
                for neigh in devices:
                    if dev!=neigh:
                        devices[dev]["beam_gains"][neigh]=compute_beam_gains(devices,dev,neigh)
                        beam_gains[repid][t][dev][neigh]=devices[dev]["beam_gains"][neigh]

            for dev in devices:
                devices[dev]["traffic"]=traffics[repid][t][dev]                                
            
            ### Compute SNRs and SINRs
            for ct in range(channel_trials): # 100 times
                signals={}
                for dev in devices:
                    signals[dev]={}
                    for neigh in devices:
                        if dev!=neigh:
                            signals[dev][neigh]=compute_link_signal(devices[neigh]["beam_gains"][dev][0],devices[dev]["beam_gains"][neigh][1], devices[dev]["distances"][neigh])
                SNRs,SINRs=compute_link_SINRS(signals,devices)

                for dev in SINRs:
                    for neigh in SINRs[dev]:
                        
                        for i in range(3): # THREE algos
                            devices[dev]["SNRs"][neigh][i][t*channel_trials+ct]=SNRs[dev][neigh][i]                        
                            devices[dev]["SINRs"][neigh][i][t*channel_trials+ct]=SINRs[dev][neigh][i]

            #### STORE
            pickle.dump(devices, open('simu/{0}_{1}_{2}_optim_analysis.dat'.format(ExpeName,Repe,t), 'wb'))
            

            ##### plot patterns if configured for
            if plot_device_patterns and plot_device_patterns_iters>t:
                plotted_beams=plot_patterns(devices,t)

            ### stop here if configured for
            if single_instance_of_pb:
                break

        #####################################################
        ####### Repe plots ##################################

        cdfplots[Repe]={}
        sumsinr_plot[Repe]={}

        figure_plots={}
        figure_plots_tot={}
        siners_array=[]
        snrs_array=[]
        for dev in devices:
            for neigh in devices:
                if dev!=neigh:
                    lele=len(cdfplots[Repe])
                    cdfplots[Repe][lele]=plot_cdf(devices[dev]["SNRs"][neigh],devices[dev]["SINRs"][neigh])
                    cdfplots[Repe][lele]["title"]="{0} to {1}, {2:.2f}m".format(neigh,dev,devices[dev]["distances"][neigh])
                    if neigh==devices[dev]["peer"]:
                        siners_array.append(devices[dev]["SINRs"][neigh])
                        snrs_array.append(devices[dev]["SNRs"][neigh])

        sumsinr_plot[Repe][0]=plot_cdf_sum_SINR(siners_array,legend_on=True)
        sumsinr_plot[Repe][0]["title"]=""#"CDF - Sum of SINRS of connected links (dB)"
        print("cdf throughput", Repe)
        sumsinr_plot[Repe][1]=plot_cdf_sum_throughputs(siners_array,legend_on=True)
        sumsinr_plot[Repe][1]["title"]=""#"CDF - Troughputs on interfered links (dB)"

        # aggregate the repe data to the global simu container
        if repid==0:
            aggregated_multi_repe["SINRs"]=siners_array
            aggregated_multi_repe["SNRs"]=snrs_array
        else:
            for dev in devices:
                aggregated_multi_repe["SINRs"][dev]=np.concatenate((aggregated_multi_repe["SINRs"][dev],siners_array[dev]),axis=1)
                aggregated_multi_repe["SNRs"][dev]=np.concatenate((aggregated_multi_repe["SNRs"][dev],snrs_array[dev]),axis=1)

        # some adjustments
        figure_plots[0]=topo[0]
        del figure_plots[0]["title"]
        figure_plots[0]["x_axis_label"]="Topology"
        del figure_plots[0]["y_axis_label"]
        del figure_plots[0]["side_bar"]

        for dev in devices:
            #54 grid
            figure_plots[0]["values"][dev]["arrowprops"]["lw"]=1.2
            figure_plots[0]["values"][nb_devices+dev]["size"]=6
            figure_plots[0]["values"][2*nb_devices+dev]['s']=25
            #33 grid
            # figure_plots[0]["values"][dev]["arrowprops"]["lw"]=1.5
            # figure_plots[0]["values"][nb_devices+dev]["size"]=12
            # figure_plots[0]["values"][2*nb_devices+dev]['s']=40

        figure_plots[1]=sumsinr_plot[Repe][0]
        # figure_plots[2]=sumsinr_plot[Repe][1]
        if  TOPO_SCENARIO=="2 para links" or TOPO_SCENARIO=="4 corners" :
            figure_plots[2]=plot_cdf_mean_SINR(snrs_array, with_interf=False, legend_on=True)
            figure_plots[3]=sumsinr_plot[Repe][1]

        else:
            figure_plots[3]=plot_cdf_total_throughputs(siners_array,legend_on=True)
            # figure_plots[4]=plot_cdf_total_throughputs(siners_array,dir_load="down",legend_on=True, xmax=5600) #24mesh
            figure_plots[4]=plot_cdf_total_throughputs(siners_array,dir_load="down",legend_on=True)

        figure_plots_tot[0]=plot_cdf_total_throughputs(siners_array)
        figure_plots_tot[1]=plot_cdf_total_throughputs(siners_array,dir_load="down")

        figure_plots_tot[2]=plot_cdf_min_throughputs(siners_array)
        figure_plots_tot[3]=plot_cdf_min_throughputs(siners_array,dir_load="down")
        figure_plots_tot[4]=plot_cdf_median_throughputs(siners_array)
        figure_plots_tot[5]=plot_cdf_median_throughputs(siners_array,dir_load="down")
        figure_plots_tot[6]=plot_cdf_max_throughputs(siners_array)
        figure_plots_tot[7]=plot_cdf_max_throughputs(siners_array,dir_load="down")     


        figure_plots_tot[8]=plot_cdf_sum_SINR(snrs_array)
        figure_plots_tot[8]["title"]=""#"CDF - Sum of SINRS of connected links (dB)"
        figure_plots_tot[9]=plot_cdf_sum_throughputs(snrs_array)
        figure_plots_tot[9]["title"]=""#"CDF - Troughputs on interfered links (dB)"       

        figure_plots_tot[10]=plot_box_SINR(siners_array)
        # figure_plots[10]=plot_box_SINR(siners_array)
        figure_plots_tot[11]=plot_box_SINR(snrs_array)
        # figure_plots[11]=plot_box_SINR(snrs_array,snr=True)
        
        figure_plots_tot[12]=plot_box_tpt(siners_array)
        # figure_plots[12]=plot_box_tpt(siners_array)
        figure_plots_tot[13]=plot_box_tpt(snrs_array)

        if len(plotted_beams)>0 and False:
            figure_plots_tot[14]=plotted_beams[3%nb_devices][0]# def
            figure_plots_tot[15]=plotted_beams[3%nb_devices][2]# opt tx
            # figure_plots_tot[15]=plotted_beams[5%nb_devices][2]# opt tx
            # figure_plots_tot[16]=plotted_beams[7%nb_devices][3]# opt rx
            figure_plots_tot[16]=plotted_beams[3%nb_devices][3]# opt rx

        if not redo_figs_ultimate:        
            figure_plots_tot[17]=plot_box_bg(beam_gains[repid], to_peer=True)
            # figure_plots[17]=plot_box_bg(beam_gains[repid], to_peer=True)
            # figure_plots[18]=plot_box_bg(beam_gains[repid], to_peer=False)
            figure_plots_tot[18]=plot_box_bg(beam_gains[repid], to_peer=False)


        # prepared_FIRST=guiguiplots.prepare_plots(cdfplots[Repe])
        # guiguiplots.plot_pages(prepared_FIRST, nb_plots_hor=3, nb_plots_vert=4, show=False, PDF_to_add=pp1)
        prepared_SUMSI=guiguiplots.prepare_plots(sumsinr_plot[Repe])
        guiguiplots.plot_pages(prepared_SUMSI, nb_plots_hor=1, nb_plots_vert=1, show=False, PDF_to_add=pp1)

        prepared_fig=guiguiplots.prepare_plots(figure_plots)
        guiguiplots.plot_pages(
            prepared_fig, nb_plots_hor=4, nb_plots_vert=5, show=False, PDF_to_add=pp1,
            user_defined_tlfs=8, user_defined_alfs=10)

        # just nicknames for topos
        nick="other"
        if  TOPO_SCENARIO=="2 para links" :
            nick="4long"
        if  TOPO_SCENARIO=="4 corners" :
            nick="4corners"
        if TOPO_SCENARIO=="Controlled_Mesh" :
            nick="{0}mesh_{1}".format(nb_devices, Repe)
        if  TOPO_SCENARIO=="BounceNet12" :
            nick="BN12-6"
            
        guiguiplots.plot_pages(
            prepared_fig, 
            nb_plots_hor=4, nb_plots_vert=5,
            grid_specs=[],
            show=False,
            user_defined_tlfs=8, user_defined_alfs=10,
            file_to_save="eps_fig_{0}_{1}".format(nick,date_simu),
            dir_to_save="eps_figs", 
            format_to_save='eps',
            user_defined_dpi=300)

        # just to crop and convert to eps valid for IEEE confs
        subprocess.run(["epstopdf", "eps_figs/eps_fig_{0}_{1}_0.eps".format(nick,date_simu)])
        subprocess.run(["pdfcrop", "eps_figs/eps_fig_{0}_{1}_0.pdf".format(nick,date_simu)])
        subprocess.run(["pdftops", "-eps", "eps_figs/eps_fig_{0}_{1}_0-crop.pdf".format(nick,date_simu), nick+".eps"])
        # subprocess.run(["pdftops", "eps_figs/eps_fig_{0}_{1}_0-crop.pdf".format(nick,date_simu), nick+".eps"])

        prepared_fig_tot=guiguiplots.prepare_plots(figure_plots_tot)
        guiguiplots.plot_pages(prepared_fig_tot, nb_plots_hor=3, nb_plots_vert=3, show=False, PDF_to_add=pp1)

        if single_instance_of_pb:
            break

    ##### Aggregated Data
    pickle.dump(objective_funcs, open('simu/{0}_{1}_{2}_{3}_objective_funcs.dat'.format(
        ExpeName,
        OPTIM_ITER_T0,
        OPTIM_PROCESS_POP,
        const_K), 'wb'))

    #####################################################
    ####### Aggregated plots ##################################    

    figure_plots_MR={}

    figure_plots_MR[0]=plot_cdf_sum_SINR(aggregated_multi_repe["SINRs"])
    figure_plots_MR[0]["title"]=""#"CDF - Sum of SINRS of connected links (dB)"
    figure_plots_MR[1]=plot_cdf_mean_SINR(aggregated_multi_repe["SINRs"])
    figure_plots_MR[1]["title"]=""#"CDF - Sum of SINRS of connected links (dB)"
    print("aggregated_multi_repe")
    figure_plots_MR[2]=plot_cdf_sum_throughputs(aggregated_multi_repe["SINRs"])
    figure_plots_MR[2]["title"]=""#"CDF - Troughputs on interfered links (dB)"

    figure_plots_MR[3]=plot_cdf_total_throughputs(aggregated_multi_repe["SINRs"])
    figure_plots_MR[4]=plot_cdf_total_throughputs(aggregated_multi_repe["SINRs"],dir_load="down")

    figure_plots_MR[5]=plot_cdf_sum_SINR(aggregated_multi_repe["SNRs"])
    figure_plots_MR[5]["title"]=""#"CDF - Sum of SINRS of connected links (dB)"
    figure_plots_MR[6]=plot_cdf_mean_SINR(aggregated_multi_repe["SNRs"], with_interf=False, legend_on=True)
    figure_plots_MR[6]["title"]=""#"CDF - Troughputs on interfered links (dB)"       
    figure_plots_MR[7]=plot_cdf_sum_throughputs(aggregated_multi_repe["SNRs"])
    figure_plots_MR[7]["title"]=""#"CDF - Troughputs on interfered links (dB)"       

    figure_plots_MR[8]=plot_cdf_total_throughputs(aggregated_multi_repe["SNRs"])
    figure_plots_MR[9]=plot_cdf_total_throughputs(aggregated_multi_repe["SNRs"],dir_load="down")

    
    prep_figs_MR=guiguiplots.prepare_plots(figure_plots_MR)
    guiguiplots.plot_pages(prep_figs_MR, nb_plots_hor=5, nb_plots_vert=4, show=False, PDF_to_add=pp1,user_defined_tlfs=6)


    pp1.close()
    optim.pp.close()
    
