#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:28:04 2019

@author: Guillaume Gaillard
"""

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import itertools
from multiprocessing import Pool, cpu_count, get_context # , set_start_method
# set_start_method("spawn")

import time
import pickle
from wrapper_plottings import plottings as guiguiplots

# from geneticalgorithm.geneticalgorithm import geneticalgorithm as ga
import os
if not os.path.exists('simu'):
    os.makedirs('simu')

############### Globals for local execution 
nb_devices=4
peers=list(range(nb_devices))[::-1]
depart=time.time()

rng = np.random.default_rng(88736)
initial_val=rng.integers(1,5, size=(nb_devices*2*32,))
set_initial_iterator_value=True                                 # random searches start at initial_val
dont_stress_on_rolls=True                                       # a phase shift on every ae is harmless. 
# so if True, random searches may roll over similar sectors (modulo a phase shift on active aes.) 
# if False, sectors with first active ae valued > 1 will be artificially discarded (may lead to init issues) 
gen_from_initial=False                                          # initial value from external source
gen_random=False                                                # start from as many random initial values as processes
im_id=0                                                         # global figure id for plottings 

###################### globals for direct calls, similar as in simu_IASSEN.py
exp_comp=-1j
log_dist=True
log_dist=False

tx_pow=100#<24dBm
lambda_carrier=0.005#m
PLE=2.2#path loss exponent
Other_losses=40
noise_mean=8*10**-8#150#10**136#.2 #20000000
noise_std_dev=10**-8
interf_coef_mean=.001
interf_coef_var=.00025
######### log distance
d0=1
PL0=10*np.log10(lambda_carrier**2/((4*np.pi)**2 * Other_losses))
Xg_sig=3.92#*0.05
Xg_sig_mean=Xg_sig
if log_dist:
    PLE=2.2#3.2#.27#path loss exponent
    interf_coef_mean=.001#0835
    interf_coef_var=.0002#.25
channel_trials=110
# HYPER-PARAMETERS
max_active_antennas=32#19#32#8#32#13#25#18         # Maximum number of active antennas
min_active_antennas=1#4#5#2#9

phase_shifts=np.array([0,-np.pi/2,np.pi,np.pi/2])

const_K=8

# DEVICES PARAMETERS
# compute random distances
dists=rng.random((nb_devices,nb_devices))*15
dists=(dists + dists.T)/2
np.fill_diagonal(dists,0)

# instanciate amps
amps=rng.uniform(low=0., high=30, size=(nb_devices,nb_devices,32))
for x in range(nb_devices):    
    amps[x,x,:]=0.

SORT_BY_AMPS=True                                  # artificially give a different shape to the objective function
SORT_BY_AMPS=False
NORMALIZE=False                                    # let amplitude characteristics be in average 1.
NORMALIZE=True

## sort and normalize
for x in range(nb_devices):
    for y in range(nb_devices):
        if x!=y:
            if NORMALIZE:
                mean=max(10**-10,np.mean(amps[x,y,:]))
            else:
                mean=1
            if SORT_BY_AMPS:
                sorted_a=np.sort(amps[x,y,:])[::-1]
            else:
                sorted_a=amps[x,y,:]

            amps[x,y,:]=sorted_a/mean

# instanciate phases, traffics, noises
phases=rng.uniform(low=-np.pi, high=np.pi, size=(nb_devices,nb_devices,32))
traffics=rng.normal(interf_coef_mean,interf_coef_var, size=(nb_devices))
noise=max(10**-10,rng.normal(noise_mean,noise_std_dev))
noises=[noise]*nb_devices
interf_coef=interf_coef_mean#max(0.0001,rng.normal(interf_coef_mean,interf_coef_var))/2

# links, interferers, main links, etc. numpyfied notation
links=np.array([a for a in itertools.permutations(range(nb_devices), 2)])
interferers=np.zeros((nb_devices,nb_devices-2),dtype='int')
dev_mlinks=np.zeros((nb_devices,),dtype='int')

for dev in range(nb_devices):
    main_link=-1
    link_id=0
    interfering_links=[]
    for link in links:
        if link[1]==dev:
            if link[0]==peers[dev]:
                dev_mlinks[dev]=link_id
            else:
                interfering_links.append(link_id)
        link_id+=1
    interferers[dev,:]=np.array(interfering_links)


#### PDF variables
plot_prefix='optim_' #'SNR_model_default'
pp = None
ExpeName = "def_exp"

################################################################################################### Gain funcs
### apply Friis or log-distance model
def compute_channel_gain(distance):
    if log_dist:
        return(10**(0.1*PL0) * 10**(-0.1*Xg_sig_mean)/(distance**PLE))
    return(lambda_carrier**2/((4*np.pi)**2 * distance**PLE * Other_losses))

### make it global
cpted_chan_gains=compute_channel_gain(dists[links[:,0],links[:,1]])

#### vectorized computation of sector gains
def v_compute_beam_gain(amplis,phases,shifts):
    
    return(
        abs(                                            # compleex modulus 
            np.sum(                                     # 
                amplis*                                 # amplitude characteristics
                np.where(shifts==0,0,1)*                # active aes (where shifts (state vals) are not null)
                np.exp(                                 # 
                    (                                   # 
                        phase_shifts[shifts-1]          # e.g. state val 3 means active with phase shift 2 (pi) 
                        -   phases                      # phase characteristics
                    )*exp_comp),                        # -j
                axis=1                                  # 
            )
        )
    )

################################################ Exploration funcs

# A wrapper to the objective function allowing to multiprocess it with initial values 
def wrap_sum_SINRS(unformatted_sec):
    
    # musect=unformatted_sec
    musect=np.fromiter(unformatted_sec,dtype='int')

    # multiply active terms (val 1) with a random 1,5 val
    if set_local_phase_random:
        rng = np.random.default_rng()
        bob=rng.integers(1,5, size=(256,))
        musect*=bob
        musect=np.mod(musect,5)

    # add an initial value modulo 5
    if set_initial_iterator_value:
        studied_sec=np.mod(initial_val+musect,5)
    else:
        studied_sec=musect

    return(sum_SINRS(studied_sec),studied_sec)

# A wrapper processing a portion of the space 
def wrap_iter_portion(mytuple):
    max_ae_id=6#16#32#3          # for each sector (32 aes), the portion includes the combinations of active aes among these first

    print("Work",mytuple,max_ae_id)
    (i,j,k,l)=mytuple
    reformatted_tuple=(i[0],i[1],j[0],j[1],k[0],k[1],l[0],l[1], max_ae_id)
    res=iter_portion(*reformatted_tuple)
    print("Work",mytuple,max_ae_id, "->",res[0])
    return(res)

# processing a portion of the space
def iter_portion(tx0, rx0, tx1, rx1, tx2, rx2, tx3, rx3, max_ae_id):
    # e.g. tx1==3 means 3 active antenna elements among the first max_ae_id aes of the 32 aes of tx sector of dev 1

    #objective : best sum of sinrs
    ssmax=0
    best_multi=np.zeros(256, dtype='int')#np.zeros((4,2,32))    

    # iterate over the combinations (tx and rx can be switched) of active aes 
    for active_set_tx_0,active_set_rx_0 in itertools.product(
            itertools.combinations(range(max_ae_id), tx0),
            itertools.combinations(range(max_ae_id), rx0),
        ) if tx0!=rx0 else itertools.combinations_with_replacement(
                itertools.combinations(range(max_ae_id), tx0), 2):
        for active_set_tx_1,active_set_rx_1 in itertools.product(
                itertools.combinations(range(max_ae_id), tx1),
                itertools.combinations(range(max_ae_id), rx1),
            ) if tx1!=rx1 else itertools.combinations_with_replacement(
                    itertools.combinations(range(max_ae_id), tx1), 2):
            # print("half",tx0, rx0, tx1, rx1, tx2, rx2, tx3, rx3)
            for active_set_tx_2,active_set_rx_2 in itertools.product(
                    itertools.combinations(range(max_ae_id), tx2),
                    itertools.combinations(range(max_ae_id), rx2),
                ) if tx2!=rx2 else itertools.combinations_with_replacement(
                        itertools.combinations(range(max_ae_id), tx2), 2):
                for active_set_tx_3,active_set_rx_3 in itertools.product(
                        itertools.combinations(range(max_ae_id), tx3),
                        itertools.combinations(range(max_ae_id), rx3),
                    ) if tx3!=rx3 else itertools.combinations_with_replacement(
                            itertools.combinations(range(max_ae_id), tx3), 2):

                    
                    config=np.zeros(256, dtype='int')

                    # first active ae of each sec has phase val = 0 (don't roll)
                    config[active_set_tx_0[0]]=1
                    config[active_set_rx_0[0]+32]=1
                    config[active_set_tx_1[0]+64]=1
                    config[active_set_rx_1[0]+96]=1
                    config[active_set_tx_2[0]+128]=1
                    config[active_set_rx_2[0]+160]=1
                    config[active_set_tx_3[0]+192]=1
                    config[active_set_rx_3[0]+224]=1
                    
                    # iterate over phase values
                    for phase_vec in itertools.product(
                        itertools.combinations_with_replacement(
                            itertools.product(range(1,5),repeat=tx0-1), 2) if tx0==rx0 else itertools.product(
                                itertools.product(range(1,5),repeat=tx0-1),
                                itertools.product(range(1,5),repeat=rx0-1)
                            ),
                        itertools.combinations_with_replacement(
                            itertools.product(range(1,5),repeat=tx1-1), 2) if tx1==rx1 else itertools.product(
                                itertools.product(range(1,5),repeat=tx1-1),
                                itertools.product(range(1,5),repeat=rx1-1)
                            ),
                        itertools.combinations_with_replacement(
                            itertools.product(range(1,5),repeat=tx2-1), 2) if tx2==rx2 else itertools.product(
                                itertools.product(range(1,5),repeat=tx2-1),
                                itertools.product(range(1,5),repeat=rx2-1)
                            ),
                        itertools.combinations_with_replacement(
                            itertools.product(range(1,5),repeat=tx3-1), 2) if tx3==rx3 else itertools.product(
                                itertools.product(range(1,5),repeat=tx3-1),
                                itertools.product(range(1,5),repeat=rx3-1)
                            )
                        ):


                        ### write config
                        lpv=[]
                        for pvi in (phase_vec):
                            for pvii in (pvi):
                                lpv+=list(pvii)
                        
                        if tx0>1:
                            config[list(active_set_tx_0)[1:]]=lpv[:tx0-1]                    
                        if rx0>1:
                            config[np.array(active_set_rx_0)[1:]+32]=lpv[tx0-1:tx0+rx0-2]
                        if tx1>1:
                            config[np.array(active_set_tx_1)[1:]+64]=lpv[tx0+rx0-2:tx0+rx0+tx1-3]
                        if rx1>1:
                            config[np.array(active_set_rx_1)[1:]+96]=lpv[tx0+rx0+tx1-3:tx0+rx0+tx1+rx1-4]
                        if tx2>1:
                            config[np.array(active_set_tx_2)[1:]+128]=lpv[tx0+rx0+tx1+rx1-4:tx0+rx0+tx1+rx1+tx2-5]
                        if rx2>1:
                            config[np.array(active_set_rx_2)[1:]+160]=lpv[tx0+rx0+tx1+rx1+tx2-5:tx0+rx0+tx1+rx1+tx2+rx2-6]
                        if tx3>1:
                            config[np.array(active_set_tx_3)[1:]+192]=lpv[tx0+rx0+tx1+rx1+tx2+rx2-6:tx0+rx0+tx1+rx1+tx2+rx2+tx3-7]
                        if rx3>1:
                            config[np.array(active_set_rx_3)[1:]+224]=lpv[tx0+rx0+tx1+rx1+tx2+rx2+tx3-7:]
                            
                        ss=sum_SINRS(config)

                        # PRINT reduces speed.
                        # print(config)
                        # print(np.array2string(np.array(config,dtype='int'),threshold=np.inf, max_line_width=np.inf).replace(" ",""))

                        if ss>ssmax:
                            ssmax=ss
                            best_multi=config[:]
                            # print(ss)

    return(ssmax,best_multi)

#### explore various portions simultaneously
def explore_protions_multiprocess():
    # portions are defined by num of active ae per sector

    active_aes=itertools.product(
        # itertools.combinations_with_replacement(range(1,3),2),
        itertools.combinations_with_replacement(range(1,6),2),
        repeat=4)   
    # active_aes=[((1,1),(1,1),(1,1),(1,1))]
    # active_aes=[((2,2),(2,2),(2,2),(2,2))]

    with get_context("fork").Pool(processes=cpu_count()-1 or 1) as p:
    # with get_context("fork").Pool(processes=1) as p:
        # ss=max(p.imap(wrap_iter_portion, active_aes,chunksize=8))
        ss=max(p.imap_unordered(wrap_iter_portion, active_aes,chunksize=8))
        # ss=max(p.map(wrap_iter_portion, active_aes))

    print(ss)
    final=time.time()

    print(depart,final,final-depart)

### plot a surface representation of a portion of the space
def plot_it(degree):
    # X two devs sectors
    # Y the other 2 devs sectors

    ## degree = number of elements trialed at the beginning of each sector
    ax_dim=degree**4
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X = np.arange(0, ax_dim, 1)
    Y = np.arange(0, ax_dim, 1)
    Xm, Ym = np.meshgrid(X, Y)

    Xconfig=np.zeros((ax_dim,256), dtype='int')
    Yconfig=np.zeros((ax_dim,256), dtype='int')

    combx=0
    for quadx in itertools.product(range(degree), repeat=4):#box:
        dirdev=0
        curi=0
        for ind in quadx: 
            curi=dirdev*32
            for ae in range(ind):
                Xconfig[combx,curi+ae]=rng.integers(1,5) # a random phase val
            dirdev+=1
        combx+=1

    comby=0
    for quady in itertools.product(range(degree), repeat=4):#boy:
        dirdev=0
        curi=128
        for ind in quady:
            curi=dirdev*32+128
            for ae in range(ind):
                Yconfig[comby,curi+ae]=rng.integers(1,5) # a random phase val
            dirdev+=1
        comby+=1

    Zinter=np.zeros((ax_dim,ax_dim,256), dtype='int')

    for px in range(ax_dim):
        for py in range(ax_dim):
            Zinter[px,py,:]=Xconfig[px]+Yconfig[py]

    ### compute the values
    Z=np.zeros((ax_dim,ax_dim))
    for px in range(ax_dim):
        for py in range(ax_dim):
            Z[px,py]=sum_SINRS(Zinter[px,py,:])

    ### find max in numpy
    maxind=np.argmax(Z)
    maxind_urv=np.unravel_index(maxind,Z.shape)

    print(maxind, maxind_urv)
    print(Z[maxind_urv])
    print(Zinter[maxind_urv])
    print(Zinter[maxind_urv].reshape(4,2,32))

    ### plot surface
    surf = ax.plot_surface(Xm, Ym, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()

    return(Zinter[maxind_urv].reshape(4,2,32))

### Iterator-based subspace explorations
def explore_subspace(subspace_iterator,plot_results=True, multiprocess=False):

    
    #objective : best sum of sinrs
    ssmax=0
    best_multi=np.zeros((4,2,32)) 

    # prepare data to show
    if plot_results:
        y=[]

    # one single process used
    if not multiprocess:
        for x in subspace_iterator:
            ss=sum_SINRS(x)
            if ss>ssmax:
                ssmax=ss
                best_multi=x
            if plot_results:
                y.append(ss)

    else:
        timeop=200000       # let's subslice portion
        chunksize=40000     # size of chunk given to each proc
        chunknum=0
        
        while True:
            print("chunknum:",chunknum)
            chunknum+=1

            sss=[]

            sector_pool_slice=itertools.islice(subspace_iterator,0,timeop)
            with get_context("fork").Pool(processes=cpu_count()-1 or 1) as p:
                sss=p.map(wrap_sum_SINRS, sector_pool_slice, chunksize=chunksize)

            # chunk has something
            if len(sss)>0:
                results=[ss[0] for ss in sss]
                maxind=np.argmax(results)
                if sss[maxind][0]>ssmax:
                    ssmax=sss[maxind][0]
                    best_multi=sss[maxind][1]
                if plot_results:
                    y+=results
            else: #last chunk is empty
                break

    print(ssmax,best_multi)

    if plot_results:
        plt.plot(y)
        plt.show()

### Wrapper to iterator based subspace explorations
def build_and_explore():
    global set_initial_iterator_value, set_local_phase_random

    # a random start
    bob=rng.integers(5, size=(256,))

    ######## INSTEAD of one portion with start at a random value, make wrapper start at 0
    set_initial_iterator_value=True
    #### shuffle phases?
    set_local_phase_random=False
    # set_local_phase_random=True

    ###########VARIATE one component in 256
    # npit=np.zeros((5,256), dtype='int')
    # for i in range(5):
    #     npit[i,:]=bob
    #     npit[i,0]=i
    # subspace_iterator=npit
    # # explorer
    # explore_subspace(subspace_iterator,plot_results=True, multiprocess=False)


    ############# ONE PORTION of the entirety
    # the universe
    sector_pool=itertools.product(range(5), repeat=32*8)
    # a slice of the universe
    subspace_iterator=itertools.islice(sector_pool,0,2000000)
    explore_subspace(subspace_iterator,plot_results=True, multiprocess=True)



    ################## EXPLORE SAME AS plot_it(5) work a bit locally
    # ranges_as=[
    #     [0,0,0,0,0]+[0]*27,
    #     [1,0,0,0,0]+[0]*27,
    #     [1,1,0,0,0]+[0]*27,
    #     [1,1,1,0,0]+[0]*27,
    #     [1,1,1,1,0]+[0]*27,
    #     [1,1,1,1,1]+[0]*27
    #     ]

    # subspace_iterator_zip=itertools.product(iter(ranges_as),repeat=8)
    # subspace_iterator=itertools.starmap(itertools.chain,subspace_iterator_zip)
    # explore_subspace(subspace_iterator,plot_results=True, multiprocess=True)

### A dispatcher for random strategies of exploration
def get_optim_sectors(strategy="random_uniform", maxiter=None, population_size=100, initial_idv=None):
    global initial_val, gen_from_initial, gen_random

    # a very simple strategy
    if strategy=="random_uniform":
        return (rng.integers(5, size=(4,2,32)))

    # use of genetic algorithm
    if strategy=="genetic":
        return (genetic(maxiter=maxiter,population_size=population_size, initial_idv=initial_idv))

    # just the plot_it func: surface of a subspace
    if strategy=="plot_simple":
        return (plot_it(5))

    # A Gibbs sampling strategy
    if strategy=="gibbs":
        # if there is an initial value to start with, directly run the algo on it (single proc)
        if not (initial_idv is None):
            gibbs_sampled=gibbs_sampler(maxiter,initial_idv)
            return(np.array(gibbs_sampled[1],dtype='int').reshape(4,2,32))
        # otherwise multiprocess pop_size Gibbs algos from different starts
        return (np.array(multi_func(maxiter,population_size,func=gibbs)[1],dtype='int').reshape(4,2,32))

    # Simulated Annealing strategy
    if strategy=="simu_anneal":
        # start from random
        gen_random=True
        # if there is an initial value to start with, directly run pop_size algos on it (pop_size processes)
        if not (initial_idv is None):
            initial_val=initial_idv
            gen_from_initial=True
        # otherwise multiprocess pop_size SA algos from different starts
        return (np.array(multi_func(maxiter,population_size,func=simu_anneal)[1],dtype='int').reshape(4,2,32))
    
    # Simulated Annealing strategy with fixed reception sectors
    if strategy=="simu_anneal_fixed_rx":
        # start from random
        gen_random=True
        # if there is an initial value to start with, directly run pop_size algos on it (pop_size processes)
        if not (initial_idv is None):
            initial_val=initial_idv
            gen_from_initial=True
        # otherwise multiprocess pop_size SA algos from different starts
        return (np.array(multi_func(maxiter,population_size,func=simu_anneal_fixed_rx)[1],dtype='int').reshape(4,2,32))

    # IASSEN strategy
    if strategy=="simu_anneal_with_pre_gibbs":
        # start from controlled 
        gen_random=False
        # if there is an initial value to start with, directly run pop_size algos on it (pop_size processes)
        if not (initial_idv is None):
            initial_val=initial_idv
            gen_from_initial=True
        # otherwise multiprocess pop_size Gibbs+SA+Gibbs algos from different starts
        return (np.array(multi_func(maxiter,population_size,func=simu_anneal_with_pre_gibbs,plot_multi_temp=True)[1],dtype='int').reshape(nb_devices,2,32))

    if strategy=="test_sectors":
        test_sectors=np.zeros((4,2,32),dtype='int')
        test_sectors[0,:]=[0,0,0,0,0,1,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4,0,0,0,0,5,0,0,0,0,1,0]
        test_sectors[1,:]=[0,0,0,1,0,0,2,0,0,3,0,0,4,0,0,5,0,0,1,0,0,2,0,0,3,0,0,4,0,0,5,0]
        test_sectors[2,:]=[0,0,0,0,1,0,0,0,2,0,0,0,3,0,0,0,4,0,0,0,5,0,0,0,1,0,0,0,2,0,0,0]
        test_sectors[3,:]=[0,0,1,0,2,0,3,0,4,0,5,0,1,0,2,0,3,0,4,0,5,0,1,0,2,0,3,0,4,0,5,0]

        print(test_sectors)
        return(test_sectors)

    return (rng.integers(5, size=(4,2,32)))

################################################ Exploration algos

### calling GA
def genetic(maxiter=None, population_size=100, initial_idv=None):

    varbound=np.array([[0,4]]*256)

    algoparams={
        'max_num_iteration': maxiter,
        'population_size':population_size,
        'mutation_probability':0.5,
        'elit_ratio': 0.01,
        'crossover_probability': 0.5,
        'parents_portion': 0.1,
        'crossover_type':'uniform',
        'max_iteration_without_improv':None,
        'multiprocessing_ncpus': 8,
        'multiprocessing_engine': None,
        }

    model=ga(function=obj_func,dimension=256,variable_type='int',variable_boundaries=varbound, algorithm_parameters=algoparams)
    model.run(initial_idv=initial_idv, plot=True)

    # print(model.report)

    return(np.array(model.output_dict['variable'],dtype='int').reshape(4,2,32))

#### Gibbs sampler
def gibbs(cycles):
    return(gibbs_sampler(cycles,generate_initial()))    
def gibbs_sampler(cycles,ori):
    # make some copies of the original state
    X=ori.copy()
    bestX=X.copy()

    # start values
    ssmax=-obj_func(X)                          # evaluate opposite of func to be minimized => maximize
    ssmaxori=ssmax
    converged=False                             # breaker 
    range_c=rng.permutation(nb_devices*2*32)    # Shuffle the antenna elements

    successive_ss=[ssmax]                       # Store values along alg. execution

    for c in range(cycles):
        X_init=bestX.copy()                     # init state is best previous state
        for i in range_c:                       # roll over the aes
            X=bestX.copy()                      # another copy
            ssmi=0                              
            for j in range(1,5):                # roll over the 4 other values
                X[i]=(X[i]+1)%5                 
                ss=-obj_func(X)
                ssmi=max(ssmi,ss)
                if ss>ssmax:                    # update if found better
                    ssmax=ss
                    bestX[i]=X[i]
            if ssmi==0:
                bestX[i]=0                      # unlock a situation where OF is always 0
                
        if np.array_equal(bestX,X_init):        # no change => local optimum reached
            converged=True
            break
        successive_ss.append(ssmax)
    print("gibbs", ssmaxori,'->',ssmax, converged, c)
    return(ssmax,bestX,c,successive_ss, list(range(cycles)))

#### Gibbs sampler with fixed rx sectors
def gibbs_fixed_rx(cycles,ori):
    X=ori.copy()
    bestX=X.copy()
    ssmax=-obj_func(X)                          # evaluate opposite of func to be minimized => maximize
    ssmaxori=ssmax
    converged=False
    range_c=rng.permutation(128)                # permutation over half the aes

    successive_ss=[ssmax]

    for c in range(cycles):
        X_init=bestX.copy()
        for iflat in range_c:
            i=(iflat//32)*64+(iflat%32)         # index within tx aes
            X=bestX.copy()
            for j in range(1,5):                # roll over the 4 other values
                X[i]=(X[i]+1)%5
                ss=-obj_func(X)
                if ss>ssmax:
                    ssmax=ss
                    bestX[i]=X[i]
        if np.array_equal(bestX,X_init):
            converged=True
            break
        successive_ss.append(ssmax)
    print("gibbs_fixed_rx", ssmaxori,'->',ssmax, converged, c)
    return(ssmax,bestX,c,successive_ss, list(range(cycles)))

#### Simulated Annealing algorithm
def simu_anneal(cycles):
    return(simu_anneal_base(cycles,generate_initial()))
def simu_anneal_base(cycles,ori):
    # Initial copies
    X=ori.copy()
    nextX=X.copy()
    bestX=X.copy()
    
    ssmin= obj_func(X)
    ssmax=-ssmin
    ssmin_ori=ssmin
    converged=False
    
    successive_ss=[ssmin]
    successive_Ts=[]                                # Values of Temperature

    # For each cycle of Alg.
    for c in range(cycles):
        X=nextX.copy()                              # update state
        T=const_K/(np.log2(2+c))                    # Compute the temperature parameter T

        next_ijs=[]                                 # store possible transitions (ae index, value)
        energies=[]                                 # store corresp energies
        for i in range(256):
            for j in range(1,5):
                X[i]=(X[i]+1)%5
                ss= obj_func(X)
                if -ss>ssmax:
                    ssmax=-ss
                    bestX=X.copy()
                if ss!=0:
                    next_ijs.append((i,j))
                    energies.append(ss-ssmin)       # gain from original situation
            X[i]=nextX[i]

        # Compute the corresponding state probability Pi_x
        NRJ_a=np.array(energies)
        inexp=np.clip(-NRJ_a/T,a_min=None, a_max=702)                   # avoid too high values
        exps_NRJs=np.exp(inexp)
        sum_energies=np.nan_to_num(sum(exps_NRJs))
        denom=np.clip(sum_energies-exps_NRJs,a_min=10**-10, a_max=None) # avoid null denoms
        Pi_xs=np.clip(np.nan_to_num(exps_NRJs/denom),a_min=0, a_max=1)  # probas in [0,1]
        Pi_xs/=np.nan_to_num(sum(Pi_xs))                                # now sum == 1

        # Sample a random variable according to law Pi(.), and choose the next state accordingly        
        try:
            next_idx=rng.choice(np.arange(len(Pi_xs)), 1, p=Pi_xs)[0]
        except:
            # something went wrong, stop
            print(T, sum_energies, min(energies), max(energies), min(exps_NRJs), max(exps_NRJs), sum(Pi_xs))
            converged=True
            break

        ### get back the stored value
        ssmin=energies[next_idx]+ssmin
        new_i,new_j=next_ijs[next_idx]
        nextX[new_i]=(nextX[new_i]+new_j)%5
        # assert(obj_func(nextX)==ssmin)

        ### store the choice
        successive_ss.append(ssmin)
        successive_Ts.append(T)

    # Output: the final state
    print("SA", ssmin_ori,'->',ssmin, converged, c)
    return(-ssmin,nextX, c, successive_ss, successive_Ts)

#### Simulated Annealing algorithm with fixed rx sectors
def simu_anneal_fixed_rx(cycles):
    return(simu_anneal_fixed_rx_base(cycles,generate_initial()))
def simu_anneal_fixed_rx_base(cycles,ori):
    X=ori.copy()
    nextX=X.copy()
    bestX=X.copy()

    ssmin= obj_func(X)                      
    ssmax=-ssmin
    ssmin_ori=ssmin
    converged=False
    
    successive_ss=[ssmin]
    successive_Ts=[]

    for c in range(cycles):
        X=nextX.copy()
        T=const_K/(np.log2(2+c))

        next_ijs=[]
        energies=[]
        for dev in range(4):
            for i in range(dev*64,dev*64+32):
                for j in range(1,5):
                    X[i]=(X[i]+1)%5
                    ss= obj_func(X)
                    if -ss>ssmax:
                        ssmax=-ss
                        bestX=X.copy()
                    if ss!=0:
                        next_ijs.append((i,j))
                        energies.append(ss-ssmin)
                X[i]=nextX[i]

        NRJ_a=np.array(energies)
        inexp=np.clip(-NRJ_a/T,a_min=None, a_max=702)
        exps_NRJs=np.exp(inexp)
        # sum_energies=np.nan_to_num(sum(exps_NRJs),posinf=np.exp(708))
        sum_energies=np.nan_to_num(sum(exps_NRJs))
        # denom=np.clip(sum_energies-exps_NRJs,a_min=10**-10, a_max=np.exp(708))
        denom=np.clip(sum_energies-exps_NRJs,a_min=10**-10, a_max=None) 
        Pi_xs=np.clip(np.nan_to_num(exps_NRJs/denom),a_min=0, a_max=1)
        Pi_xs/=np.nan_to_num(sum(Pi_xs))                                

        try:
            next_idx=rng.choice(np.arange(len(Pi_xs)), 1, p=Pi_xs)[0]
        except:
            # something went wrong, stop
            print(T, sum_energies, min(energies), max(energies), min(exps_NRJs), max(exps_NRJs), sum(Pi_xs))
            converged=True
            break

        ssmin=energies[next_idx]+ssmin
        new_i,new_j=next_ijs[next_idx]
        nextX[new_i]=(nextX[new_i]+new_j)%5
        # assert(obj_func(nextX)==ssmin)

        successive_ss.append(ssmin)
        successive_Ts.append(T)

    print("SA fixed rx", ssmin_ori,'->',ssmin, converged, c)
    return(-ssmin,nextX, c, successive_ss, successive_Ts)

#### Simulated Annealing algorithm wrapped with Gibbs phases (IASSEN)
def simu_anneal_with_pre_gibbs(cycles):
    return(simu_anneal_pre_gibbs(cycles,generate_initial()))
def simu_anneal_pre_gibbs(cycles,ori):

    # Start with a Gibbs phase
    bob=gibbs_sampler(cycles,ori)
    successive_ss=[-i for i in bob[3]]
    successive_Ts=[const_K]*(len(successive_ss)-1)
    ssmax,bestX=bob[0],bob[1]

    X=bestX.copy()
    nextX=X.copy()
    ssmin= obj_func(X)                      
    ssmax=-ssmin
    ssmin_ori=ssmin
    converged=False

    for c in range(cycles):
        X=nextX.copy()
        T=const_K/(np.log2(2+c))

        next_ijs=[]
        energies=[]
        for i in range(nb_devices*2*32):
            for j in range(1,5):
                X[i]=(X[i]+1)%5
                ss= obj_func(X)
                if -ss>ssmax:
                    ssmax=-ss
                    bestX=X.copy()
                if ss!=0:
                    next_ijs.append((i,j))
                    energies.append(ss-ssmin)
            X[i]=nextX[i]

        NRJ_a=np.array(energies)
        if sum(NRJ_a)==0:                                       # No transition improves, we are at an optimum => random choice
            new_i=rng.integers(nb_devices*2*32)
            new_j=rng.integers(1,5)
            nextX[new_i]=(nextX[new_i]+new_j)%5
        else:
            inexp=np.clip(-NRJ_a/T,a_min=None, a_max=600)
            exps_NRJs=np.exp(inexp)
            sum_energies=np.nan_to_num(sum(exps_NRJs))
            denom=np.clip(sum_energies-exps_NRJs,a_min=10**-10, a_max=None)
            Pi_xs=np.clip(np.nan_to_num(exps_NRJs/denom),a_min=0, a_max=1)
            Pi_xs/=np.nan_to_num(sum(Pi_xs))

            try:
                next_idx=rng.choice(np.arange(len(Pi_xs)), 1, p=Pi_xs)[0]
            except:
                # something went wrong, stop
                print(T, sum_energies, min(energies), max(energies), min(exps_NRJs), max(exps_NRJs), sum(Pi_xs))
                converged=True
                break

            ssmin=energies[next_idx]+ssmin
            new_i,new_j=next_ijs[next_idx]
            nextX[new_i]=(nextX[new_i]+new_j)%5
            # assert(obj_func(nextX)==ssmin)

        successive_ss.append(ssmin)
        successive_Ts.append(T)

    print("SA", ssmin_ori,'->',ssmin, converged, c)

    ### Post Gibbs phase
    bob=gibbs_sampler(cycles,bestX)
    successive_ss+=[-i for i in bob[3]]
    successive_Ts.append(T)
    ssmax,bestX=bob[0],bob[1]

    print("IASSEN", -ssmin_ori,'->',ssmax, converged, c)
    return(ssmax,bestX, c, successive_ss, successive_Ts)

#### Simulated Annealing algorithm with fixed rx sectors wrapped with Gibbs phases with fixed rx sectors
def simu_anneal_with_pre_gibbs_fixed_rx(cycles):
    return(simu_anneal_pre_gibbs_fixed_rx(cycles,generate_initial()))
def simu_anneal_pre_gibbs_fixed_rx(cycles,ori):

    bob=gibbs_fixed_rx(cycles,ori)
    successive_ss=[-i for i in bob[3]]
    successive_Ts=[const_K]*(len(successive_ss)-1)
    ssmax,bestX=bob[0],bob[1]

    X=bestX.copy()
    nextX=X.copy()
    ssmin= obj_func(X)
    ssmax=-ssmin
    ssmin_ori=ssmin
    converged=False

    for c in range(cycles):
        X=nextX.copy()
        T=const_K/(np.log2(2+c))#-0.01*c

        next_ijs=[]
        energies=[]
        for dev in range(4):
            for i in range(dev*64,dev*64+32):
                for j in range(1,5):
                    X[i]=(X[i]+1)%5
                    ss= obj_func(X)
                    if -ss>ssmax:
                        ssmax=-ss
                        bestX=X.copy()
                    if ss!=0:
                        next_ijs.append((i,j))
                        energies.append(ss-ssmin)
                X[i]=nextX[i]

        NRJ_a=np.array(energies)
        inexp=np.clip(-NRJ_a/T,a_min=None, a_max=702)
        exps_NRJs=np.exp(inexp)
        sum_energies=np.nan_to_num(sum(exps_NRJs),posinf=np.exp(708))
        denom=np.clip(sum_energies-exps_NRJs,a_min=10**-10, a_max=np.exp(708))
        Pi_xs=np.clip(np.nan_to_num(exps_NRJs/denom),a_min=0, a_max=1)
        Pi_xs/=np.nan_to_num(sum(Pi_xs))                                
        
        try:
            next_idx=rng.choice(np.arange(len(Pi_xs)), 1, p=Pi_xs)[0]
        except:
            print(T, sum_energies, min(energies), max(energies), min(exps_NRJs), max(exps_NRJs), sum(Pi_xs))
            converged=True
            break

        ssmin=energies[next_idx]+ssmin
        new_i,new_j=next_ijs[next_idx]
        nextX[new_i]=(nextX[new_i]+new_j)%5
        # assert(obj_func(nextX)==ssmin)

        successive_ss.append(ssmin)
        successive_Ts.append(T)

    print("SA fixed rx", ssmin_ori,'->',ssmin, converged, c)

    ### Post Gibbs phase
    bob=gibbs_fixed_rx(cycles,bestX)
    successive_ss+=[-i for i in bob[3]]
    successive_Ts.append(T)
    ssmax,bestX=bob[0],bob[1]

    print("G+SA+G fixed rx", -ssmin_ori,'->',ssmax, converged, c)
    return(ssmax,bestX, c, successive_ss, successive_Ts)


########################################################################################################################    
#############################################  OBJECTIVE FUNCTIONS  ####################################################

# First version of objective function, not fully vectorized
# Only for 4 devs
# Shows literal expression of sum of SINRs
def sum_SINRS_old(musect):
    multisector_test=np.array(musect,dtype='int').reshape(nb_devices,2,32)
    rssis=np.clip(
        tx_pow
        *v_compute_beam_gain(amps[links[:,0],links[:,1],:],phases[links[:,0],links[:,1],:],multisector_test[links[:,0],0,:])
        *cpted_chan_gains
        *v_compute_beam_gain(amps[links[:,1],links[:,0],:],phases[links[:,1],links[:,0],:],multisector_test[links[:,1],1,:])
        ,
        a_min=10**-10, a_max=None
    )
    return(
        rssis[2]/(interf_coef/2*(rssis[5]+rssis[8])+noises[3])
        +rssis[4]/(interf_coef/2*(rssis[1]+rssis[11])+noises[2])
        +rssis[9]/(interf_coef/2*(rssis[3]+rssis[6])+noises[0])
        +rssis[7]/(interf_coef/2*(rssis[0]+rssis[10])+noises[1])
    )

# Objective function 
def sum_SINRS(musect):
    # numpify if needed
    multisector_test=np.array(musect,dtype='int').reshape(nb_devices,2,32)
    # vector version of RSSI formula
    rssis=np.clip(
        tx_pow
        *v_compute_beam_gain(amps[links[:,0],links[:,1],:],phases[links[:,0],links[:,1],:],multisector_test[links[:,0],0,:])
        *cpted_chan_gains
        *v_compute_beam_gain(amps[links[:,1],links[:,0],:],phases[links[:,1],links[:,0],:],multisector_test[links[:,1],1,:])
        ,
        a_min=10**-10, a_max=None
    )

    # vector version of SINR formula
    return(
        np.sum(
            rssis[dev_mlinks]/
            (
                # 1/2*
                (
                    np.sum(traffics[links[interferers,0]]*rssis[interferers],1)
                )+
                noises
            )
        )
    )

# Objective function decored to exclude redondant cases
def obj_func(X):
    # min active antenna elements is not met globally
    if sum(X)<min_active_antennas*nb_devices*2:
        return(0)
    for i in range(nb_devices*2):
        Xdev=X[32*i:32*(i+1)]
        # max active ae locally overcome
        if len(np.nonzero(Xdev)[0])>max_active_antennas:
            return(0)
        # min active ae not met locally
        if sum(Xdev)<min_active_antennas:
            return(0)
        # if configured so, check only one of the 4 shifts of the sector (the one with first active ae valued 1)
        if not dont_stress_on_rolls:
            for j in Xdev:
                if j!=0:
                    if j==1:
                        break
                    return(0)

    # if no problem then eval normally
    return(-sum_SINRS(X))

########################################################################################################################    
########################################################################################################################    

################################################ Exploration Helpers

## Dispatcher to multiple functions, gathers results from multiple processes
def multi_func(cycles,nb_origins,func=gibbs, plot_multi_temp=False):
    # measure start time
    dep=time.time()

    # list of results of multiprocessing pool
    list_bathed=[]

    cpus=cpu_count()-1 or 1
    # with get_context("spawn").Pool(processes=cpus) as p:
    # with Pool(processes=cpus) as p:
    with get_context("fork").Pool(processes=cpus) as p:
        back_from_the_pool=p.imap_unordered(func,[cycles]*nb_origins)
        list_bathed=list(back_from_the_pool) # 
        # p.close()

    # just get the list of result values
    list_vals=[lb[0] for lb in list_bathed]
    mm=np.argmax(list_vals)

    # plot the exploration process
    plot_bathed(list_bathed,plot_multi_temp=plot_multi_temp, func=func) ### if func is SA, each result has a temperature function (can be plotted or not)

    # final best result
    return(list_bathed[mm])

### plotting the exploration process
def plot_bathed(list_bathed, plot_multi_temp=False, func=gibbs):
    global im_id, pp

    # check if there is a PDF to plot to
    if pp is None:
        pp = guiguiplots.PdfPages(plot_prefix+"Main_ESS_optim.pdf")

    # store data, to plot again if necessary
    pickle.dump(list_bathed, open('simu/{0}_{1}_optim_cycles.dat'.format(ExpeName,im_id), 'wb'))

    # case of one temperature line
    if (not plot_multi_temp) and (not func==gibbs):
        # take the longest temp line
        last_temp=np.argmax([len(lb[-1]) for lb in list_bathed])

        plot_data={}
        plot_data[0]={
            "values":{
                0:{
                    "type":"plot",
                    "y_values":list_bathed[last_temp][-1],
                    "x_values":range(1,len(list_bathed[last_temp][-1])+1),
                    "linestyle":'--', 'legend':"temperature", 'color':'k'},
            },
            "title":"SA run {0}".format(im_id),
            "legends":{},
        }

        # per process results
        i=0
        plot_data[0]["twinx_values"]={}
        for res_ss in list_bathed:
            i+=1
            plot_data[0]["twinx_values"][i]={
                    "type":"plot",
                    "y_values":res_ss[-2],
                    'legend':"obj func {}".format(i)}

        # plotting
        multi_proc_plots=guiguiplots.prepare_plots(plot_data)
        guiguiplots.plot_pages(multi_proc_plots, nb_plots_hor=1, nb_plots_vert=1, grid_specs=[], show=False, PDF_to_add=pp)

    else: # SA one temp., or gibbs
        plot_data={}
        plot_data[0]={
            "title":"", #"SA run {0}".format(im_id),
            "legends":{"legend_labels_font_size":16},
            "tight_layout":False
        }

        i=0
        plot_data[0]["values"]={}
        plot_data[0]["twinx_values"]={}
        for res_ss in list_bathed:
            i+=1
            # temp
            if (not func==gibbs): # case of SA, one temperature line per proc
                plot_data[0]["values"][i]={
                        "type":"plot",
                        "y_values":res_ss[-1],
                        "linestyle":'--',
                        "linewidth":1.2,
                        }
                if i==3: # choose one for legend
                    plot_data[0]["values"][i]['legend']="Temperature {}".format(i)
            
            # per proc obj func
            plot_data[0]["twinx_values"][i]={
                    "type":"plot",
                    "y_values":res_ss[-2],
                    "linewidth":1.2,
                    }
            if i==3:
                plot_data[0]["twinx_values"][i]['legend']="Objective function {}".format(i)
            
        multi_proc_plots=guiguiplots.prepare_plots(plot_data)
        guiguiplots.plot_pages(multi_proc_plots, nb_plots_hor=1, nb_plots_vert=1, grid_specs=[], show=False, PDF_to_add=pp,user_defined_tlfs=16)

    im_id+=1

####### Build initial situation the algos depart from
def generate_initial():
    global rng
    rng = np.random.default_rng()               # re-seed for multiprocess (each fork will get different val)
    if gen_from_initial:                        # use the one already prepared
        return(initial_val)
    if gen_random:                              # a new random state
        return(rng.integers(5, size=(nb_devices*2*32,)))        

    # if not any of the previous two, then build one with minimal active aes
    actives=np.array([rng.permutation(np.arange(32*i, 32*(i+1)))[:min_active_antennas] for i in range(nb_devices*2)]).flatten()
    X=np.zeros(nb_devices*2*32, dtype='int')
    X[actives]=1
    return(X)


########################################################################## Update funcs

#### External func to update global variables of the problem
def update_np(devices,amps_from, phases_from,config_dict):
    global depart,log_dist,tx_pow,lambda_carrier,PLE,Other_losses,d0,PL0,Xg_sig_mean
    global max_active_antennas,min_active_antennas,phase_shifts
    global dists,amps,phases,links,cpted_chan_gains,traffics
    global interf_coef,noises
    global const_K
    global nb_devices, peers, interferers, dev_mlinks

    nb_devices=len(devices)
    peers=config_dict["peers"]
    const_K=config_dict["const_K"]
    depart=time.time()
    log_dist=config_dict["log_dist"]
    tx_pow=config_dict["tx_pow"]
    lambda_carrier=config_dict["lambda_carrier"]
    PLE=config_dict["PLE"]
    Other_losses=config_dict["Other_losses"]
    interf_coef=config_dict["interf_coef_mean"]
    d0=config_dict["d0"]
    PL0=config_dict["PL0"]
    Xg_sig_mean=config_dict["Xg_sig_mean"]
    max_active_antennas=config_dict["max_active_antennas"]
    min_active_antennas=config_dict["min_active_antennas"]
    phase_shifts=config_dict["phase_shifts"]
    noises=config_dict["noises"]
    traffics=config_dict["interf_coefs"]
    dists=np.zeros((nb_devices,nb_devices))

    for dev in devices:
        for neigh in devices:
            if dev!=neigh:
                dists[dev,neigh]=devices[dev]["distances"][neigh]    

    links=np.array([a for a in itertools.permutations(range(nb_devices), 2)])
    interferers=np.zeros((nb_devices,nb_devices-2),dtype='int')
    dev_mlinks=np.zeros((nb_devices,),dtype='int')

    for dev in range(nb_devices):
        main_link=-1
        link_id=0
        interfering_links=[]
        for link in links:
            if link[1]==dev:
                if link[0]==peers[dev]:
                    dev_mlinks[dev]=link_id
                else:
                    interfering_links.append(link_id)
            link_id+=1
        interferers[dev,:]=np.array(interfering_links)        

    cpted_chan_gains=compute_channel_gain(dists[links[:,0],links[:,1]])

    amps=amps_from
    phases=phases_from

#### External func to update a subset of global variables of the problem
def update_amps_and_phases_np(devices,amps_from, phases_from, traffics_from):
    global dists,amps,phases,traffics

    amps=amps_from
    phases=phases_from
    traffics=traffics_from

if __name__ == '__main__':
    ## Trials iter portion
    # iter_portion(1, 1, 1, 1, 1, 1, 1, 1, 32)
    # iter_portion(1, 1, 1, 1, 1, 1, 1, 4, 32)
    # iter_portion(1, 1, 1, 1, 1, 1, 5, 7, 31)
    # iter_portion(1, 1, 1, 1, 1, 1, 1, 8, 32)
    # iter_portion(1, 1, 1, 1, 1, 1, 8, 8, 32) #12G RAM
    # iter_portion(7, 7, 7, 7, 7, 7, 7, 7, 32) #4G RAM

    ## Explore portions of space 
    # explore_protions_multiprocess()

    ## plot a surface representation of a portion of the space
    # plot_it(5)

    ## iterator-based explorations
    # build_and_explore()

    ###### Random-based explorations
    ### Try one randomly
    # res=get_optim_sectors(strategy="random_uniform", maxiter=None, population_size=100, initial_idv=None)
    ### Try one parametrically
    # res=get_optim_sectors(strategy="test_sectors")
    # a random state
    bob=rng.integers(1,5, size=(256,))
    
    ## use a genetic algorithm from a random position 
    ####### requires GA (easy way: "git clone https://github.com/Guillaumegaillard/geneticalgorithm.git")  
    ####### which in turn requires func-timeout python module (pip install)
    # res=get_optim_sectors(strategy="genetic", maxiter=1000, population_size=100, initial_idv=bob)
    # res=get_optim_sectors(strategy="gibbs", maxiter=1000, population_size=100, initial_idv=bob)
    # res=get_optim_sectors(strategy="gibbs", maxiter=1000, population_size=10)
    res=get_optim_sectors(strategy="simu_anneal", maxiter=50, population_size=7, initial_idv=bob)
    # res=get_optim_sectors(strategy="simu_anneal", maxiter=50, population_size=7)
    # res=get_optim_sectors(strategy="simu_anneal_fixed_rx", maxiter=50, population_size=7)
    # res=get_optim_sectors(strategy="simu_anneal_with_pre_gibbs", maxiter=50, population_size=7)

    print(res)


    # bob=multi_func(20)
    # print(bob)

    if not (pp is None):
        pp.close()