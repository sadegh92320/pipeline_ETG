import pandas as pd
import os
from math import sqrt
from datetime import datetime, timedelta
import csv
import json
import math
from convert_gaze import convert
import pytz
import numpy as np

import traceback
import pickle
import gzip
from eye_traking_analysis import fixation_AOI
from ultralytics import YOLO

class UtilitiesETG:
    def __init__(self):
        pass
    def saccade_amplitude(self, f):
        sacc = False
        point = 0
        distances = []        
        for index, line in f.iterrows():     
            if line["eye_movement"] == "saccade" and sacc == False:
                point = (float(line["gaze2d_x"]), float(line["gaze2d_y"]))
                sacc = True
            if line["eye_movement"] == "fixation" and sacc == True:
                    
                distance = math.sqrt((float(line["gaze2d_x"]) - point[0])**2 + (line["gaze2d_y"] - point[1])**2)
                    
                if not math.isnan(distance):
                    distances.append(distance)
                sacc = False  
        avg = sum(distances)/len(distances)
        return avg

    def saccade_velocity(self, f):
        sacc = False
        point = 0
        velocities = []
        
        for index, line in f.iterrows():
           
            if line["eye_movement"] == "saccade" and sacc == False:
                point = (float(line["gaze2d_x"]), float(line["gaze2d_y"]))
                time = float(line["timestamp"])
                sacc = True
            if line["eye_movement"] == "fixation" and sacc == True:
                
                distance = math.sqrt((float(line["gaze2d_x"]) - point[0])**2 + (line["gaze2d_y"] - point[1])**2)
                velocity = distance/(float(line["timestamp"] - time))
                
                if not math.isnan(velocity):
                    velocities.append(velocity)
                sacc = False
        avg = sum(velocities)/len(velocities)
        return avg
    
    def duration_fixation(self, f):
        fix = False
        time = 0
        times = []
        
        for index, line in f.iterrows():
           
            if line["eye_movement"] == "fixation" and fix == False:
                time = line["timestamp"]
                fix = True
            if line["eye_movement"] == "saccade" and fix == True:
                times.append(float(line["timestamp"]) - float(time))
                fix = False
        
        avg = sum(times)/len(times)
       
        return avg
    
class UtilitiesDrivingData:
    def __init__(self):
        pass
    def get_velocity(self, f):
        total = 0
        nb = 0
        for i in range(1,len(f)-1):
            try:
                velocity = sqrt((float(f[" Velocity x"][i])*float(f[" Velocity x"][i])) + (float(f[" Velocity z"][i])*float(f[" Velocity z"][i])))
                if velocity > 16 or velocity < 4:
                    continue
                else:
                    total += velocity
                    nb += 1
            except:
                continue
        try:
            return (total/nb)
        except:
            return 0
        
    def get_steering(self, f):
        g = []
        for i in range(3,len(f)-1):
            try:
                gradient = (float(f[" Steering"][i])-float(f[" Steering"][i-3]))/((float(f[" Time"][i]) - float(f[" Time"][i-3])))
                g.append(gradient)
            except:
                continue
        change = 0
        for i in range(1, len(g) - 1):
            if ((g[i-1] > 0 and g[i] < 0) or (g[i-1] < 0 and g[i] > 0)):
                change = change + 1 
        return change
    
    def get_brake(self, f):
        g = []
        for i in range(3,len(f)-1):
            try:
                gradient = (float(f[" Brake"][i])-float(f[" Brake"][i-3]))/((float(f[" Time"][i]) - float(f[" Time"][i-3])))
                g.append(gradient)
            except:
                continue
        change = 0
        for i in range(1, len(g) - 1):
            if ((g[i-1] > 0 and g[i] < 0) or (g[i-1] < 0 and g[i] > 0)):
                change = change + 1 
        return change
    
    def get_steer_velocity(self, f):
        total = 0
        nb = 0
        
        for i in range(3,len(f)-1):
            try:
                gradient = abs(((float(f[" Steering"][i]) - float(f[" Steering"][i-3]))*90) / (float(f[" Time"][i]) - float(f[" Time"][i-3])))
                total += gradient
                nb += 1
            except:
                continue
        try:
            return total/nb
        except:
            return 0
        
    def get_number_of_colision(self, f):
        return f[" CollidedWithBystander"].iloc[-2]
    
    def has_collided_pedes_or_bus(self, f):
        return f[" CollidedWithTarget"].iloc[-2]
        
    def get_max_steer_angle(self, f):
        series = pd.to_numeric(f[" Steering"], errors='coerce').dropna()
        series = series.astype(float)
        return series.max()
    
    def get_velocity_std(self, f):
        f[' Velocity x'] = pd.to_numeric(f[' Velocity x'], errors='coerce')
        f[' Velocity y'] = pd.to_numeric(f[' Velocity y'], errors='coerce')
        f[' Velocity z'] = pd.to_numeric(f[' Velocity z'], errors='coerce')

        f['total_velocity'] = np.sqrt(
            f[' Velocity x']**2 + f[' Velocity y']**2 + f[' Velocity z']**2
        )
        filtered = f[(f['total_velocity'] >= 3) & (f['total_velocity'] <= 17)]

        std_dev = filtered['total_velocity'].std()
        return std_dev

    def get_max_min_acceleration(self, f):
        acc = []
        for i in range(3,len(f)-1):

            try:
                velocity1 = sqrt((float(f[" Velocity x"][i])*float(f[" Velocity x"][i])) + (float(f[" Velocity z"][i])*float(f[" Velocity z"][i])))
                velocity2 = sqrt((float(f[" Velocity x"][i-3])*float(f[" Velocity x"][i-3])) + (float(f[" Velocity z"][i-3])*float(f[" Velocity z"][i-3])))
                if velocity1 > 16 or velocity2 > 16:
                    continue
                gradient = ((velocity1 - velocity2) / (float(f[" Time"][i]) - float(f[" Time"][i-3])))
                acc.append(gradient)
            except: 
                continue
        return (max(acc),min(acc))
    
    def get_accident_speed(self, f):
        try:
            speed = f.loc[(f[' CollidedWithTarget']).str.strip() == "True"]
            if len(speed) > 1:
                line = speed.iloc[0]
                velocity = sqrt((float(line[" Velocity x"])*float(line[" Velocity x"])) + (float(line[" Velocity z"])*float(line[" Velocity z"])))
                return velocity
            else:
                return "NA"
        except:
            return "No collide with target for this test"
        

    def get_reaction_time(self, f):
        #print(f[' PedSpawned'])
        pedscenario = [5,6,7,8]
        if int(f[" SceneNr"][0]) not in pedscenario:  
                    return "NA"
        
        t_ped = f.loc[(f[' PedSpawned']).str.strip() == "True", ' Time']
        
        if len(t_ped) > 1:
            t_collision = f.loc[(f[' CollidedWithTarget']).str.strip() == "True", ' Time']
            t_ped = t_ped.iloc[0]
            start_i = f[f[' PedSpawned'].str.strip() == "True"].index[0]
            sub_df = f.loc[start_i:]
            sub_df.loc[:, ' Brake'] = pd.to_numeric(sub_df[' Brake'], errors='coerce').fillna(0)

            t_brake = sub_df.loc[sub_df[' Brake'] > 0, " Time"]     
            if len(t_brake) > 1:
                if len(t_collision) > 0 and t_collision.iloc[0] > t_brake.iloc[0]:
                    return -1
                return float(t_brake.iloc[0]) - float(t_ped)
            else: 
                return -1
        else:
            return 0
        
    def diff_steer(self, part_nb, steer, scenario_number, data):
        if scenario_number in [2,3,4]:
            steer_1 = data[(part_nb,1)]["Number of Steering"]
            return ((steer - steer_1)/steer_1)*100
        else:
            return "NA"
    
    def diff_brake(self, part_nb, brake, scenario_number, data):
        if scenario_number in [2,3,4]:
            brake_1 = data[(part_nb,1)]["Number of Braking"]
            try:
                return ((brake - brake_1)/brake_1)*100
            except:
                return 100
        else:
            return "NA"
        
    def diff_avg_vel(self, part_nb, avg_vel, scenario_number, data):
        if scenario_number in [2,3,4]:
            avg_vel_1 = data[(part_nb,1)]["Average Velocity (m-per-s)"]
            return ((avg_vel - avg_vel_1)/avg_vel_1)*100
        else:
            return "NA"
    
    def diff_steer_vel(self, part_nb, steer_vel, scenario_number, data):
        if scenario_number in [2,3,4]:
            steer_vel_1 = data[(part_nb,1)]["Steering Velocity"]
            return ((steer_vel - steer_vel_1)/steer_vel_1)*100
        else:
            return "NA"
    
    def diff_steer_max(self, part_nb, max_steer, scenario_number, data):
        if scenario_number in [2,3,4]:
            max_steer_1 = data[(part_nb,1)]["Max Steering Angle"]
            return ((max_steer - max_steer_1)/max_steer_1)*100
        return "NA"
    
    def diff_decc(self, part_nb, max_decc, scenario_number, data):
        if scenario_number in [2,3,4]:
            max_decc_1 = data[(part_nb,1)]["Max Deceleration (m-per-s^2)"]
            return ((max_decc - max_decc_1)/max_decc_1)*100
        else:
            "NA"
        
    def diff_acc(self, part_nb, max_acc, scenario_number, data):
        if scenario_number in [2,3,4]:
            max_acc_1 = data[(part_nb,1)]["Max Acceleration (m-per-s^2)"]
            return ((max_acc - max_acc_1)/max_acc_1)*100
        else:
            return "NA"
        
    def diff_stand_vel(self, part_nb, stand_vel, scenario_number, data):
        if scenario_number in [2,3,4]:
            stand_dev_1 = data[(part_nb,1)]['Standard Deviation Velocity (m-per-s)']
            return ((stand_vel - stand_dev_1)/stand_dev_1)*100
        else:
            return "NA"
    
