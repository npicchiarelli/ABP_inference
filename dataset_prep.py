'''
Prepares the dataset for the model training.'''

import itertools
import pathlib
import pickle as pkl

import numpy as np
import pandas as pd

def data_prep(dir: str, datestamp: str, timestep: float, R: float, datafname: str):
    #Directory defining
    sim_folder = pathlib.Path(dir)
    datafolder = sim_folder / datestamp / "data"

    #Directory reading and storing in dataframes
    path_list = []
    for i,path in enumerate(datafolder.iterdir()):
        if path.is_dir():
            df_path = path / f"{datestamp}_run{i+1}.txt"
            path_list.append(df_path)

    df_list = []
    for path in path_list:
        df_list.append(pd.read_csv(path))
        
    #Velocity calculation and NaN dropping
    for df in df_list:
        df[["vx","vy"]] = df.groupby("N")[["xpos","ypos"]].diff()/timestep #velocity calculation
        df.dropna(inplace=True)
        df["Time"]=df["Time"]-1

    #Velocity and position reshaping
    num_exp = len(df_list)
    Np = df_list[0].N.max()
    num_steps = df_list[0].Time.max()

    ppos = []
    vv = []
    for df in df_list:
        Np = df.N.max()
        x = np.array(df.xpos).reshape(Np,-1)
        y = np.array(df.ypos).reshape(Np,-1)
        xy = np.stack([x,y],axis=2)
        ppos.append(xy)

        vx = np.array(df.vx).reshape(Np,-1)
        vy = np.array(df.vy).reshape(Np,-1)
        vxy = np.stack([vx,vy],axis=2)
        vv.append(vxy)

    position = np.stack(ppos,axis=3)
    position = np.transpose(position,(3,1,0,2))

    vel = np.stack(vv,axis=3)
    vel = np.transpose(vel,(3,1,0,2))

    #Drag calculation
    gamma = 6*np.pi*R*1e-3
    gamma = np.repeat(gamma, Np)
    gamma = gamma[:,np.newaxis]
    gamma = np.tile(gamma,(num_exp,num_steps,1,1))

    #Edges
    edges = list(itertools.combinations(range(Np), 2))
    edges = np.array(edges)
    
    data = {
    "position": position,
    "velocity": vel,
    "drag_coefficient": gamma,
    "edge_list": edges,}

    with open(datafname+'.pkl', '+rb') as f:
        pkl.dump(data, f)
