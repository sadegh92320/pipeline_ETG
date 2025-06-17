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
from datapreprocess import DataPreprocess



class GazeDataManager:
    def __init__(self):
        pass
    def convert_gmt_time(self, dir):
        time1 = datetime.strptime(dir, "%Y%m%dT%H%M%SZ")
                
        gmt = pytz.timezone("GMT")
        uk_time = pytz.timezone("Europe/London")  # Automatically handles DST

        # Localize the time to GMT
        time1_gmt = gmt.localize(time1)
        time1_uk = time1_gmt.astimezone(uk_time)


        # Convert to UK time
        time1 = datetime(
                year=time1_uk.year,
                month=time1_uk.month,
                day=time1_uk.day,
                hour=time1_uk.hour,
                minute=time1_uk.minute,
                second=time1_uk.second
            )
        
        return time1
    
    def retrieve_event_data(self, event_path):
        event_data = []
        with gzip.open(event_path, 'rt', encoding='utf-8') as file:  # 'rt' mode for reading text
             for line in file:
                  if line.strip():
                       if (json.loads(line.strip()))["type"] == "event":
                           event_data.append(json.loads(line.strip()))
        return event_data
    
    def extract_participant_event_info(self, event_participant, next_event_participant):
        part_number = event_participant["data"]["object"]["participantNumber"]
        scenario_number = event_participant["data"]["object"]["scenarioNumber"]
        scenario_event = event_participant["data"]["object"]["event"]
        scenario_next_event = next_event_participant["data"]["object"]["event"]
        return part_number, scenario_number, scenario_event, scenario_next_event
    
    def check_data_proceed(self, scenario_number, scenario_event, scenario_next_event):
        if scenario_number != '0':
            if scenario_event == 'ScenarioStart':
                if scenario_next_event == 'EventStart':
                    return True
        return False

    
    def get_start_end_time(self, event_data, next_event, participants, eye_tracker_data, scenario_number, part_number):
        start_log = 0
        event_time = 0
        for nb, date, files, start_time, event_time in participants:
            if nb == part_number:
                if int(files[" SceneNr"][0]) == int(scenario_number):
                    if start_time != None and event_time != None:
                        start_log = start_time
                        event_log = event_time
        if start_log == 0 or event_log == 0:
            start_scene = event_data["timestamp"]
            end_scene = next_event["timestamp"] + 10
        else:
            end_scene = next_event["timestamp"] + 10
            start_scene = end_scene - (event_log - start_log) - 10                
        i_1 = 0
        i_2 = 0
        for index, line in eye_tracker_data.iterrows():
            if line["timestamp"] > start_scene and i_1 == 0:
                i_1 = index
            if line["timestamp"] > end_scene and i_2 == 0:
                i_2 = index
                break
        return start_scene, end_scene, i_1, i_2
    
    def create_participant_data(self, data_eye, part_number, scenario_number, results_AOI):
        amplitude = self.etg_data_processor.saccade_amplitude(data_eye)
        velocity = self.etg_data_processor.saccade_velocity(data_eye)
        duration = self.etg_data_processor.duration_fixation(data_eye)
        
        result_participant  = {
                    'Participant number': part_number,
                    'Scenario number': scenario_number,
                    'Saccade amplitude': amplitude,
                    'Saccade velocity': velocity,
                    'Duration fixation': duration,
                    'fixation in side mirror': sum([result_AOI[0]["side mirror"] for result_AOI in results_AOI]),
                    'fixation in rear mirror': sum([result_AOI[0]["reer mirror"] for result_AOI in results_AOI]),
                    'fixation in speed': sum([result_AOI[0]["speed"] for result_AOI in results_AOI]),
                    'duration fixation in side mirror': sum([result_AOI[1]["side mirror"] for result_AOI in results_AOI]),
                    'duration fixation in rear mirror': sum([result_AOI[1]["reer mirror"] for result_AOI in results_AOI]),
                    'duration fixation in speed': sum([result_AOI[1]["speed"] for result_AOI in results_AOI]),
                    'number speed in view': sum([result_AOI[2]["side mirror"] for result_AOI in results_AOI]),
                    'number rear mirror in view': sum([result_AOI[2]["reer mirror"] for result_AOI in results_AOI]),
                    'number side mirror in view': sum([result_AOI[2]["speed"] for result_AOI in results_AOI]),

                }
        return result_participant
        
    def get_AOI_results(self, duration, end_scene, start_scene, eye_tracker_data, video, part_number, scenario_number, AOIs):
        duration = end_scene - start_scene
        number_AOI_trim = math.ceil(duration/120)
        results_AOI = []
        step = (end_scene - start_scene) / number_AOI_trim
        first_time = start_scene

        for i in range(number_AOI_trim):
            second_time = first_time + step
            result_AOI = fixation_AOI(eye_tracker_data, video, [first_time, second_time], part_number, scenario_number, AOIs, number=i+1)
            results_AOI.append(result_AOI)
            first_time = second_time
        return results_AOI