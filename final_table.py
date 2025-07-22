
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
from gazedatamanager import GazeDataManager
from video_manager import VideoManager
from participant_info import ParticipantInfo
from driving_data_manager import DrivingDataManager


#folder = r"\Users\Student\Desktop\driving_data_test"
#J'ai fait le 12 Avril
folder = "/Volumes/Elements/U17CC/DrivingDataLoughborough"
participants = []



class YOLOTrain:
    def __init__(self):
        pass
    def train_YOLO(self, config_path):
        # Load a model
        model = YOLO("yolo11n.pt")

        # Train the model
        train_results = model.train(
            data=config_path,  # path to dataset YAML
            epochs=100,  # number of training epochs
            imgsz=640,  # training image size
            device="cpu",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        )

        # Evaluate model performance on the validation set
        metrics = model.val()

        # Perform object detection on an image
        results = model("test_img.png")
        results[0].show()

        # Export the model to ONNX format
        path = model.export(format="onnx")  # return path to exported model

    


    
class TableConstructor:
    def __init__(self, folder = None, participant_info = None, folder_eye = None, pickel = None, AOIs = None):
        self.f = folder
        self.eye_data = folder_eye

        #to move
        self.AOI = AOIs
        self.driving_data_processor = UtilitiesDrivingData()
        self.gaze_data_manager = GazeDataManager()
        self.video_manager = VideoManager()
        self.participant_info = ParticipantInfo(participant_info)
        self.driving_data_manager = DrivingDataManager()
        self.data_preprocess = DataPreprocess(folder_eye)
        self.participants = self.get_participants()
        self.data = {}
        self.pickel = pickel
    
        
        self.df = self.get_df()
        self.df_eye = self.get_participants_gaze()
    

    def get_participants_gaze(self):
        """This method will take care of creating the table with all ETG data 
            for each participants.

            Return:
                df: pandas dataframe containing all the ETG.
        """

        #Initialize the list containing the ETG data
        final_data = []

        #Loop through all eye tracking folder in the main folder containing all ETG participants
        for root, dirs, files in os.walk(self.eye_data):
            #Loop through each directory in each folder
            for dir in dirs:
                if dir == "meta":
                    continue
                #Retrieve the GMT time at which the ETG data was recorded
                time1 = self.gaze_data_manager.convert_gmt_time(dir)

                #Retrieve the path to imudata, gazedata, eventdata
                participant_path = os.path.join(root, dir, "meta", "participant")
                imu_path = os.path.join(root, dir, "imudata.gz")
                gaze_path = os.path.join(root, dir, "gazedata.gz")
                event_path = os.path.join(root, dir, "eventdata.gz")

                #Retrieve all recorded events
                event_data = self.gaze_data_manager.retrieve_event_data(event_path)

                #Retrive the path to the ETG video
                video = os.path.join(root, dir, "scenevideo.mp4")
                #Use imu data and gaze data to create a pandas dataframe with all eye tracking data
                try:
                    data_eye = convert(imu_path, gaze_path)
                except:
                    print(f"Error processing {participant_path}")
                    continue

                #Only keeping the eye tracking data (can be changed depending on needs)
                eye_tracker_data = data_eye.loc[data_eye['label'] == "eye tracker"].reset_index(drop=True)
                
                #Some eye traking data starts at some odd time so if it is the case reset at 0
                if int(eye_tracker_data["timestamp"][0]) > 1:
                    eye_tracker_data["timestamp"] = eye_tracker_data["timestamp"] - eye_tracker_data["timestamp"][0]

                #Loop through all event retrieved previously
                for d in range(len(event_data)):
                    
                    #Retrive the participant number, the scene number, the type of event and the event coming after for each
                    # event previously stored 
                    part_number, scene_number, scenario_event, scenario_next_event = self.gaze_data_manager.extract_participant_event_info(event_data[d], event_data[d+1])
                    #Check if the current event is start of a successfull scenario
                    if self.gaze_data_manager.check_data_proceed(scene_number, scenario_event, scenario_next_event):
                                
                                #Get the start and end of the scenario in the eye tracking video, as well as the corresponding index
                                # in the pandas dataframe storing eye traking data
                                start_scene, end_scene, i_1, i_2 = self.gaze_data_manager.get_start_end_time(event_data[d], event_data[d+1], self.participants, eye_tracker_data, scene_number, part_number)

                                #Retrive all AOI statistics
                                results_AOI = self.gaze_data_manager.get_AOI_results(end_scene, start_scene, eye_tracker_data, video, part_number, scene_number, self.AOI)

                                                           
                                
                                data_eye_trimmed = eye_tracker_data[i_1:i_2]

                                result_participant = self.gaze_data_manager.create_participant_data(data_eye_trimmed, part_number, scene_number, results_AOI)
                                
                                final_data.append(result_participant)
                                

        df = pd.DataFrame(final_data)

        return df
                                    
                    
    def get_participants(self):
        """Retrieve participants information from log data.
        """
        participants = []
        #Loop through all log data
        for filename in sorted(os.listdir(self.f)):
            
            f = os.path.join(self.f, filename)

            #If the file is very small then discard 
            if os.path.isfile(f) and os.path.getsize(f) >= 50 * 1024: 
                pass
            else:
                continue
            #Retrieve the date and participant number based on the file name
            moving = filename.split("_")     
            part_number = moving[0]
            date = moving[2].replace(".csv", "")
            formatted_date = datetime.strptime(date, "%Y%m%d").strftime("%d-%m-%Y")

            try:
                
                file = pd.read_csv(f)
                #Get the end of the file based on the event trigger
                spawn_indice = self.driving_data_manager.end_of_scenario_on_spawn(file, part_number)

                collision = self.driving_data_manager.end_of_scenario_on_collision(file)

                first_non_zero, last_non_zero = self.driving_data_manager.get_begin_end_driving(file, part_number)
                                
                trimmed_df = self.data_preprocess.trim_driving_data(file, collision, first_non_zero, spawn_indice, last_non_zero)


                difference_time, time_spawn = self.driving_data_manager.get_start_scenario_event_time(file, first_non_zero)
                
                if not trimmed_df.empty:
                    participants.append((part_number, formatted_date, trimmed_df, difference_time, time_spawn))
            except Exception as e:
                
                traceback.print_exc()

                print(e)
                
                
                continue
        
        return participants
              

    def get_df(self):
        # Loop through your data and build the rows
       
        
        for nb, date, files, dummy, dummy_2 in self.participants:
            
           
            if int(files[" SceneNr"][0]) != 0:           
                key = (nb, int(files[" SceneNr"][0]))  
               
                if key in self.data and self.data[key]["Date"] > date:
                    continue
                                        
                if int(nb) == 15 or int(nb) == 56:
                    continue
            
                self.data[key] = {
                        'Date': date,
                        'Participant number': nb,
                        'Scenario number': int(files[" SceneNr"][0]),
                        'Number of Steering': self.driving_data_processor.get_steering(files),
                        'Number of Braking': self.driving_data_processor.get_brake(files),
                        'Average Velocity (m-per-s)': self.driving_data_processor.get_velocity(files),
                        'Standard Deviation Velocity (m-per-s)': self.driving_data_processor.get_velocity_std(files),
                        'Steering Velocity': self.driving_data_processor.get_steer_velocity(files),
                        'Max Steering Angle': self.driving_data_processor.get_max_steer_angle(files),
                        'Max Deceleration (m-per-s^2)': self.driving_data_processor.get_max_min_acceleration(files)[1],
                        'Max Acceleration (m-per-s^2)': self.driving_data_processor.get_max_min_acceleration(files)[0],
                        'Reaction Time (s)': self.driving_data_processor.get_reaction_time(files),
                        'Collided During Scenario': self.driving_data_processor.has_collided_pedes_or_bus(files),
                        'Number of Collision': self.driving_data_processor.get_number_of_colision(files),
                        'Speed of Accident (m-per-s)': self.driving_data_processor.get_accident_speed(files),
                        'Number of Steering Difference (%)': self.driving_data_processor.diff_steer(nb, self.driving_data_processor.get_steering(files), int(files[" SceneNr"][0]), self.data),
                        'Number of Braking Difference (%)': self.driving_data_processor.diff_brake(nb, self.driving_data_processor.get_brake(files), int(files[" SceneNr"][0]), self.data),
                        'Average Velocity Difference (%)': self.driving_data_processor.diff_avg_vel(nb, self.driving_data_processor.get_velocity(files), int(files[" SceneNr"][0]), self.data),
                        'Steering Velocity Difference (%)': self.driving_data_processor.diff_steer_vel(nb, self.driving_data_processor.get_steer_velocity(files), int(files[" SceneNr"][0]), self.data),
                        'Max Steering Difference (%)': self.driving_data_processor.diff_steer_max(nb, self.driving_data_processor.get_max_steer_angle(files), int(files[" SceneNr"][0]), self.data),
                        'Max Decelration Difference (%)': self.driving_data_processor.diff_decc(nb, self.driving_data_processor.get_max_min_acceleration(files)[1], int(files[" SceneNr"][0]), self.data),
                        'Max Acceleration Difference (%)': self.driving_data_processor.diff_acc(nb, self.driving_data_processor.get_max_min_acceleration(files)[0], int(files[" SceneNr"][0]), self.data),
                        'Standard Deviation Velocity Difference (%)': self.driving_data_processor.diff_stand_vel(nb, self.driving_data_processor.get_velocity_std(files), int(files[" SceneNr"][0]), self.data),
                        "Age": self.participant_info.get_age(nb),
                        "Gender": self.participant_info.get_gender(nb),
                    }
            
        
        if self.pickel != None:
            with open(self.pickel, 'rb') as file:
                df = pd.read_pickle(file)

            
            new_df = pd.DataFrame(self.data.values())

            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = pd.DataFrame(self.data.values())
       
        
        return df
    def generate_pickle(self):

        self.df.to_pickle("participant_result.pkl")
    
    def generate_csv(self):
        self.df.to_csv('participant_result.csv', index=False)
        self.df_eye.to_csv('particpants_result_eye.csv', index=False)
    def combine_data(self):
        merged_df = pd.merge(
                        self.df,
                        self.df_eye,
                        on=["Participant number", "Scenario number"],
                        how="inner"
                    )
        merged_df.to_csv("combined_data.csv", index=False)


#folder = "Scenarios_Charity/test"
part_info = "participant_info_3.xlsx"
#print(convert("../eye_tracking_participant/20241030T091636Z/imudata.gz", "../eye_tracking_participant/20241030T091636Z/gazedata.gz"))
#pkl_file = "participant_result.pkl"
t = TableConstructor(folder = folder, participant_info = "/Users/sadeghemami/Downloads/Form Responses 2025-06-17.csv",folder_eye="/Volumes/Elements/U17CC/ETG_Loughborough", AOIs={0:"speed", 1:"reer mirror", 2:"side mirror"})
#t.get_participants_gaze()
#pickel="participant_result.pkl"
#r"\Users\Student\Desktop\U17ccetg"

#print(t.eye_data)
#(t.get_particition_video())
#print(t.participants_eye)
#t.get_df_eye()
#(t.get_particition_video())
#print(t.data_eye)
#t.saccade_velocity()
#print(t.df.shape)

#t.generate_csv() 
#t.generate_pickle()
t.generate_csv()
