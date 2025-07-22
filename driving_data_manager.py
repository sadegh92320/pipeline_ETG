
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


class DrivingDataManager:
    def __init__(self):
        pass
    def end_of_scenario_on_spawn(self, file, part_number):
        """Get the end of the file based on the event trigger.

            Returns:
                spawn_indice (int): index corresponding to 10sec after the
                    event trigger, 0 if the event is never trigerred 
        """

        #Retrieve all indices where the event is triggered
        condition = (file[' CarSpawned'] == " True") | (file[' PedSpawned'] == " True")            
        true_indices = file.index[condition].tolist()
        #Check if the list is empty
        if len(true_indices) >0:
            #If the event was triggered then add 10sec to the index
            spawn_indice = true_indices[0] + 110
            return spawn_indice
        else:
            print(part_number)
            print(file[" SceneNr"][0])
            return 0
        
    def end_of_scenario_on_collision(self, file):
        collision = 0
        condition = (file[' CollidedWithTarget'] == " True")           
        true_indices = file.index[condition].tolist()
        if len(true_indices) > 0:
            collision = true_indices[0] + 53
        return collision
    
    def get_begin_end_driving(self, file, part_number):
        first_non_zero = 0
        last_non_zero = 0
        file[' Velocity z'] = pd.to_numeric(file[' Velocity z'], errors='coerce').fillna(0)
        non_zero_indices = file[file[' Throttle'] != 0].index
        non_zero_velo = file[file[' Velocity z'] > 1].index
        if len(non_zero_velo) > 0:
            first_non_zero = non_zero_velo[0]
        else:
            first_non_zero = 0
            print(part_number)
                
        try:
            last_non_zero = non_zero_indices[-1]
        except:
            print(part_number)
            print(file[" SceneNr"][0])
        return first_non_zero, last_non_zero
    
    def get_start_scenario_event_time(self, file, first_non_zero):
        try:
            difference_time = (file[" Time"][first_non_zero])
        except:
            difference_time = None
        try:
            time_spawn = file[(file[' PedSpawned'] == " True") | (file[' CarSpawned'] == " True")][" Time"].iloc[0]
        except:
            time_spawn = None
        
        return difference_time, time_spawn

            