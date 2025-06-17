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
from utilities import UtilitiesDrivingData, UtilitiesETG


class DataPreprocess:
    def __init__(self, eye_data):
        self.eye_data = eye_data

    def check_participants_events(self):
        for root, dirs, files in os.walk(self.eye_data):
            for dir in dirs:
                # Build the correct path to meta/participant
                if dir == "meta":
                    continue
                event_path = os.path.join(root, dir, "eventdata.gz")
                data_event = []
                with gzip.open(event_path, 'rt', encoding='utf-8') as file:  # 'rt' mode for reading text
                    for line in file:
                        # Skip empty lines or whitespace
                        if line.strip():
                            if (json.loads(line.strip()))["type"] == "event":
                                data_event.append(json.loads(line.strip()))
            scenarios_participant = set()
            for d in range(len(data_event)):
                    if data_event[d]["data"]["object"]["scenarioNumber"] != '0':
                        if data_event[d]["data"]["object"]["event"] == 'ScenarioStart':
                            if data_event[d+1]["data"]["object"]["event"] == 'EventStart':
                                scenario_number = data_event[d]["data"]["object"]["scenarioNumber"]
                                part_number = data_event[d]["data"]["object"]["participantNumber"]
                                scenarios_participant.add(int(scenario_number))
            
            if len(scenarios_participant) < 8:
                print(f"participant {part_number} has {8 - len(scenarios_participant)} missing scenario")

    def trim_driving_data(self, file, collision, first_non_zero, spawn_indice, last_non_zero):
        if collision != 0:
            trimmed_df = file.iloc[first_non_zero:collision].reset_index(drop=True)
                    
        elif spawn_indice != 0 and not np.isnan(spawn_indice):
            trimmed_df = file.iloc[first_non_zero:spawn_indice].reset_index(drop=True)
        else:
            trimmed_df = file.iloc[first_non_zero:last_non_zero].reset_index(drop=True)

        return trimmed_df

    def add_spawn(self, df):
        scene_nb = int(df[" SceneNr"][0])
        df[' Time'] = pd.to_numeric(df[' Time'], errors='coerce')
        df[' Time'] = df[' Time'] - df[' Time'].iloc[0]  # Subtract the first time value

        # Calculate time difference (delta_time)
        df['delta_time'] = df[' Time'].diff().fillna(0)  # First value is zero

        # Calculate position based on velocity and delta_time
        df[' Velocity z'] = pd.to_numeric(df[' Velocity z'], errors='coerce')
        df[' Velocity x'] = pd.to_numeric(df[' Velocity x'], errors='coerce')
        df[' Velocity y'] = pd.to_numeric(df[' Velocity y'], errors='coerce')
        df['pos x'] = (df[' Velocity x'] * df['delta_time']).cumsum()
        df['pos y'] = (df[' Velocity y'] * df['delta_time']).cumsum()
        
        
        df['pos z'] = (df[' Velocity z'] * df['delta_time']).cumsum()
        spawn_index = 0

        if scene_nb in [1,2,3,4]:
            spawn_index = df[df['pos z'] >= 801].index.min() + 86
            if pd.notnull(spawn_index):
                df.loc[spawn_index:, ' CarSpawned'] = True
        if scene_nb == 5:
            spawn_index = df[df['pos x'] >= 514].index.min() + 86
            if pd.notnull(spawn_index):
                df.loc[spawn_index:, ' PedSpawned'] = True
        if scene_nb == 6:
            spawn_index = df[df['pos x'] >= 460].index.min() + 86
            if pd.notnull(spawn_index):
                df.loc[spawn_index:, ' PedSpawned'] = True
        if scene_nb == 7:
            spawn_index = df[(df['pos z'] > 553) & (df['pos x'] > 435)].index.min() + 86
            if pd.notnull(spawn_index):
                df.loc[spawn_index:, ' PedSpawned'] = True

        if scene_nb == 8:
            spawn_index = df[df['pos z'] >= 555].index.min() + 86
            if pd.notnull(spawn_index):
                df.loc[spawn_index:, ' PedSpawned'] = True

        return spawn_index
