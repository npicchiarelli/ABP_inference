'''
Prepares the dataset for the model training.'''

import itertools
import pathlib
import pickle as pkl

import numpy as np
import pandas as pd

def data_prep(dir: str, datestamp: str, timestep: float, measevery: float, L:float, R: float, datafname: str):
    #Directory defining
    sim_folder = pathlib.Path(dir)
    datafolder = sim_folder / datestamp / "data"

    #Directory reading and storing in dataframes
    path_list = []
    for path in datafolder.iterdir():
        if path.is_dir():
            num = path.name[3:]
            df_path = path / f"{datestamp}_run{num}.txt"
            path_list.append(df_path)

    df_list = []
    for path in path_list:
        df_list.append(pd.read_csv(path))
        
    # Taking downsampling into account
    times = np.unique(df_list[0].Time)
    downsampling_factor = times[1] - times[0]
    timestep *= downsampling_factor*measevery

    #Velocity calculation and NaN dropping
    for df in df_list:
        df.sort_values(by=["N","Time"],inplace=True)
        df[["vx","vy"]] = df.groupby("N")[["xpos","ypos"]].diff()/timestep #velocity calculation
        #Accounting for PBC
        df.loc[df["vx"].abs() > (L-R)/timestep, 'vx'] -= np.sign(df[df["vx"].abs() > (L-R)/timestep].vx)*(L/timestep)
        df.loc[df["vy"].abs() > (L-R)/timestep, 'vy'] -= np.sign(df[df["vy"].abs() > (L-R)/timestep].vy)*(L/timestep)

        df[["ax","ay"]] = df.groupby("N")[["vx","vy"]].diff()/timestep #acceleration calculation
        df.dropna(inplace=True)
        df["Time"]=df["Time"]-(2*downsampling_factor)

    #Velocity, position, force, angle  reshaping
    num_exp = len(df_list)
    Np = df_list[0].N.max()
    num_steps = len(np.unique(df_list[0].Time))

    ppos = []
    vv = []
    angle_ = []
    force_ = []
    acc_ = []

    for df in df_list:
        Np = df.N.max()
        x = np.array(df.xpos).reshape(Np,-1)
        y = np.array(df.ypos).reshape(Np,-1)
        xy = np.stack([x,y],axis=2)
        ppos.append(xy)

        fx = np.array(df.fx).reshape(Np,-1)
        fy = np.array(df.fy).reshape(Np,-1)
        fxy = np.stack([fx,fy],axis=2)
        force_.append(fxy)

        vx = np.array(df.vx).reshape(Np,-1)
        vy = np.array(df.vy).reshape(Np,-1)
        vxy = np.stack([vx,vy],axis=2)
        vv.append(vxy)

        ax = np.array(df.ax).reshape(Np,-1)
        ay = np.array(df.ay).reshape(Np,-1)
        axy = np.stack([ax,ay],axis=2)
        acc_.append(axy)

        theta = np.array(df.orientation).reshape(Np,-1)
        angle_.append(theta)

    position = np.stack(ppos,axis=3)
    position = np.transpose(position,(3,1,0,2))

    force = np.stack(force_,axis=3)
    force = np.transpose(force,(3,1,0,2))

    vel = np.stack(vv,axis=3)
    vel = np.transpose(vel,(3,1,0,2))

    acc = np.stack(acc_,axis=3)
    acc = np.transpose(acc,(3,1,0,2))

    angle = np.stack(angle_, axis=2)[:,np.newaxis]
    angle = np.transpose(angle,(3,2,0,1))

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
    "acceleration": acc,
    "orientation":angle,
    "drag_coefficient": gamma,
    "node_force": force,
    "edge_list": edges,
    "dt": timestep
    }

    fdata = datafname+'.pkl'
    pathlib.Path(fdata).touch()

    with open(fdata, '+rb') as f:
        pkl.dump(data, f)

def data_prep_nosave(dir: str, datestamp: str, timestep: float, measevery: float, L:float, R: float):
    #Directory defining
    sim_folder = pathlib.Path(dir)
    datafolder = sim_folder / datestamp / "data"

    #Directory reading and storing in dataframes
    path_list = []
    for path in datafolder.iterdir():
        if path.is_dir():
            num = path.name[3:]
            df_path = path / f"{datestamp}_run{num}.txt"
            path_list.append(df_path)

    df_list = []
    for path in path_list:
        df_list.append(pd.read_csv(path))
        
    # Taking downsampling into account
    times = np.unique(df_list[0].Time)
    downsampling_factor = times[1] - times[0]
    timestep *= downsampling_factor*measevery

    #Velocity calculation and NaN dropping
    for df in df_list:
        df.sort_values(by=["N","Time"],inplace=True)
        df[["vx","vy"]] = df.groupby("N")[["xpos","ypos"]].diff()/timestep #velocity calculation
        df[["ax","ay"]] = df.groupby("N")[["vx","vy"]].diff()/timestep #acceleration calculation
        # Accounting for PBC
        df.loc[df["vx"].abs() > (L-R)/timestep, 'vx'] -= np.sign(df[df["vx"].abs() > (L-R)/timestep].vx)*(L/timestep)
        df.loc[df["vy"].abs() > (L-R)/timestep, 'vy'] -= np.sign(df[df["vy"].abs() > (L-R)/timestep].vy)*(L/timestep)
        df.dropna(inplace=True)
        df["Time"]=df["Time"]-(2*downsampling_factor)

    #Velocity, position, force, angle  reshaping
    num_exp = len(df_list)
    Np = df_list[0].N.max()
    num_steps = len(np.unique(df_list[0].Time))

    ppos = []
    vv = []
    angle_ = []
    force_ = []
    acc_ = []

    for df in df_list:
        Np = df.N.max()
        x = np.array(df.xpos).reshape(Np,-1)
        y = np.array(df.ypos).reshape(Np,-1)
        xy = np.stack([x,y],axis=2)
        ppos.append(xy)

        fx = np.array(df.fx).reshape(Np,-1)
        fy = np.array(df.fy).reshape(Np,-1)
        fxy = np.stack([fx,fy],axis=2)
        force_.append(fxy)

        vx = np.array(df.vx).reshape(Np,-1)
        vy = np.array(df.vy).reshape(Np,-1)
        vxy = np.stack([vx,vy],axis=2)
        vv.append(vxy)

        ax = np.array(df.ax).reshape(Np,-1)
        ay = np.array(df.ay).reshape(Np,-1)
        axy = np.stack([ax,ay],axis=2)
        acc_.append(axy)

        theta = np.array(df.orientation).reshape(Np,-1)
        angle_.append(theta)

    position = np.stack(ppos,axis=3)
    position = np.transpose(position,(3,1,0,2))

    force = np.stack(force_,axis=3)
    force = np.transpose(force,(3,1,0,2))

    vel = np.stack(vv,axis=3)
    vel = np.transpose(vel,(3,1,0,2))

    acc = np.stack(acc_,axis=3)
    acc = np.transpose(acc,(3,1,0,2))

    angle = np.stack(angle_, axis=2)[:,np.newaxis]
    angle = np.transpose(angle,(3,2,0,1))

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
    "acceleration": acc,
    "orientation":angle,
    "drag_coefficient": gamma,
    "node_force": force,
    "edge_list": edges,
    "dt": timestep
    }

    return data
