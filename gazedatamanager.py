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
        self.etg_data_processor = UtilitiesETG()
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
        """ From the eventdata of the eye tracking data, only the data
            labeled as event as being kept.

            Returns:
                event_data (list): List containing all recorded events from the 
                    eye tracking data.
        """
        event_data = []
        #Loop through eventdata and keepin only event
        with gzip.open(event_path, 'rt', encoding='utf-8') as file:  # 'rt' mode for reading text
             for line in file:
                  if line.strip():
                       if (json.loads(line.strip()))["type"] == "event":
                           event_data.append(json.loads(line.strip()))
        return event_data
    
    def extract_participant_event_info(self, event_participant, next_event_participant):
        """ Extract the information of interest of a recorded event to then define if the scenario
            is a successfull one.

            Returns:
                part_number (str): participant number
                scenario (str): scenario number
                scenario_event (str): the type of event that was recorded
                scenario_next_event (str): the type of event that recorded after the
                    current one
        """
        part_number = event_participant["data"]["object"]["participantNumber"]
        scenario_number = event_participant["data"]["object"]["scenarioNumber"]
        scenario_event = event_participant["data"]["object"]["event"]
        scenario_next_event = next_event_participant["data"]["object"]["event"]
        return part_number, scenario_number, scenario_event, scenario_next_event
    
    def check_data_proceed(self, scenario_number, scenario_event, scenario_next_event):
        """ Check if the event analysed is the start of a successfull scenario.

            Returns:
                Boolean: True if the event corresponds to the start of a successful scenario
                    False otherwise.
        """
        if scenario_number != '0':
            if scenario_event == 'ScenarioStart':
                if scenario_next_event == 'EventStart':
                    return True
        return False

    
    def get_start_end_time(self, event_data, next_event, participants, eye_tracker_data, scenario_number, part_number):
        """Retrieve the start and end of the scenario using the time collected in the eye tracking
            data and also the time from the log data of the corresponding participant and scenario

            Returns:
                start_scene (float): the time at which the scenario of interest starts
                    in the video
                end_scene (float): the time at which the scenario of interest ends in 
                    the video
                i_1 (int): the index in the eye tracking dataframe corresponding to the
                    start time
                i_2 (int): the index in the eye tracking dataframe corresponding to the
                    end time
        """
        start_log = 0
        event_time = 0
        #Loop through the stored participants
        for nb, date, files, start_time, event_time in participants:
            #Check if the participant and scenario number corresponds to the participant and
            # scenario being analysed
            if nb == part_number:
                if int(files[" SceneNr"][0]) == int(scenario_number):
                    if start_time != None and event_time != None:
                        #Retrieve the start time and event time in the log data
                        start_log = start_time
                        event_log = event_time
        #If it's zero (should not happen) assign the eye tracking time
        if start_log == 0 or event_log == 0:
            start_scene = event_data["timestamp"]
            end_scene = next_event["timestamp"] + 10
        else:
            #The end of the scenario corresponds to the time at which the 
            # event happens + a pre-defined time
            end_scene = next_event["timestamp"] + 10
            
            #The start of the scenario corresponds to the time at which the event happens
            # in the video minus the total time between the start and event time in the log 
            # data
            start_scene = next_event["timestamp"] - (event_log - start_log)                
        i_1 = 0
        i_2 = 0
        #In the eye tracking data find the index corresponding to the
        # time previously retrieved
        for index, line in eye_tracker_data.iterrows():
            if line["timestamp"] > start_scene and i_1 == 0:
                i_1 = index
            if line["timestamp"] > end_scene and i_2 == 0:
                i_2 = index
                break
        return start_scene, end_scene, i_1, i_2
    
    def create_participant_data(self, data_eye, part_number, scenario_number, results_AOI):
        """ Create a row of the final with all the data corresponding to a unique pair of 
            participant/scenario.

            Returns:
                result_participant (dict): result of eye tracking data of a participant/scenario pair
        """
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
        
    def get_AOI_results(self, end_scene, start_scene, eye_tracker_data, video, part_number, scenario_number, AOIs):
        """Get all AOI related data for a scenario of interest.

            Returns:
                results_AOI (list[list]): list containing all AOI data
        """
        #Get the total duration of the scenario
        duration = end_scene - start_scene
        #Check if the scenario is over 2min if yes we need to 
        # divide it in sub pieces as some gpu would crash
        number_AOI_trim = math.ceil(duration/120)
        results_AOI = []
        #Retrieve the step for each increment of piece
        step = (end_scene - start_scene) / number_AOI_trim
        first_time = start_scene

        #Loop trough the number of piece we need in the video and get the 
        # AOI results for each piece
        for i in range(number_AOI_trim):
            second_time = first_time + step
            result_AOI = fixation_AOI(eye_tracker_data, video, [first_time, second_time], part_number, scenario_number, AOIs, number=i+1)
            results_AOI.append(result_AOI)
            first_time = second_time
        return results_AOI