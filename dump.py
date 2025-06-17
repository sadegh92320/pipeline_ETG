def get_participants_gaze(self):
        final_data = []
        for root, dirs, files in os.walk(self.eye_data):
            for dir in dirs:
                # Build the correct path to meta/participant
                if dir == "meta":
                    continue
                
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
             
            
                participant_path = os.path.join(root, dir, "meta", "participant")
                if os.path.isfile(participant_path):
                    with open(participant_path, 'r') as f:
                        data = json.load(f)    
                        number = data['name']
                imu_path = os.path.join(root, dir, "imudata.gz")
                gaze_path = os.path.join(root, dir, "gazedata.gz")
                event_path = os.path.join(root, dir, "eventdata.gz")
                data_start = []
                with gzip.open(event_path, 'rt', encoding='utf-8') as file:  # 'rt' mode for reading text
                    for line in file:
                        # Skip empty lines or whitespace
                        if line.strip():
                            if (json.loads(line.strip()))["type"] == "event":
                                data_start.append(json.loads(line.strip()))
                video = os.path.join(root, dir, "scenevideo.mp4")
                try:
                    data_eye = convert(imu_path, gaze_path)
                except:
                    print(f"Error processing {participant_path}")
                    continue
                eye_tracker_data = data_eye.loc[data_eye['label'] == "eye tracker"].reset_index(drop=True)
                
                if int(eye_tracker_data["timestamp"][0]) > 1:
                    eye_tracker_data["timestamp"] = eye_tracker_data["timestamp"] - eye_tracker_data["timestamp"][0]

                #participants_eye.append([number, data_eye, eye_tracker_data, time1, video])
                divide = [6,7]
                for d in range(len(data_start)):
                    start_log = 0
                    event_time = 0
                    if data_start[d]["data"]["object"]["scenarioNumber"] != '0':
                        if data_start[d]["data"]["object"]["event"] == 'ScenarioStart':
                            if data_start[d+1]["data"]["object"]["event"] == 'EventStart':
                                part_number = data_start[d]["data"]["object"]["participantNumber"]
                                scene_number = data_start[d]["data"]["object"]["scenarioNumber"]

                                #(part_number, formatted_date, trimmed_df, difference_time)
                                for nb, date, files, start_time, event_time in self.participants:
                                    if nb == part_number:
                                        if int(files[" SceneNr"][0]) == int(scene_number):
                                            if start_time != None and event_time != None:
                                                start_log = start_time
                                                event_log = event_time
                                                print(start_log)
                                                print(event_log)
                                           



                                if start_log == 0 or event_log == 0:
                                    print("zero start time")
                                    print(part_number)
                                    print(scene_number)
                                    start_scene = data_start[d]["timestamp"]
                                    end_scene = data_start[d+1]["timestamp"] + 10
                                else:
                                    end_scene = data_start[d+1]["timestamp"] + 10
                                    start_scene = end_scene - (event_log - start_log) - 10
                                    print(end_scene)
                                    print(start_scene)
                                
                                i_1 = 0
                                i_2 = 0
                                for index, line in eye_tracker_data.iterrows():
                                        if line["timestamp"] > start_scene and i_1 == 0:
                                            
                                            i_1 = index
                    
                                        
                                        if line["timestamp"] > end_scene and i_2 == 0:
                                            
                                            
                                            i_2 = index
                                            break
                                if int(scene_number) not in divide:
                                    
                                   
                                    result_AOI = fixation_AOI(eye_tracker_data, video, [start_scene, end_scene], part_number,scene_number, self.AOI)

                                    amplitude = self.etg_data_processor.saccade_amplitude(eye_tracker_data[i_1:i_2])
                                    velocity = self.etg_data_processor.saccade_velocity(eye_tracker_data[i_1:i_2])
                                    duration = self.etg_data_processor.duration_fixation(eye_tracker_data[i_1:i_2])
                                    result_participant  = {
                                            'Participant number': part_number,
                                            'Scenario number': scene_number,
                                            'Saccade amplitude': amplitude,
                                            'Saccade velocity': velocity,
                                            'Duration fixation': duration,
                                            'fixation in side mirror': result_AOI[0]["side mirror"],
                                            'fixation in rear mirror': result_AOI[0]["reer mirror"],
                                            'fixation in speed': result_AOI[0]["speed"],
                                            'duration fixation in side mirror': result_AOI[1]["side mirror"],
                                            'duration fixation in rear mirror': result_AOI[1]["reer mirror"],
                                            'duration fixation in speed': result_AOI[1]["speed"],
                                            'number speed in view': result_AOI[2]["speed"],
                                            'number rear mirror in view': result_AOI[2]["reer mirror"],
                                            'number side mirror in view': result_AOI[2]["side mirror"],

                                        }
                                
                                
                                    final_data.append(result_participant)
                                else:
                                    
                        
                                    middle = (start_scene + end_scene)/2
                                   

                                    result_AOI = fixation_AOI(eye_tracker_data, video, [start_scene, middle], part_number, scene_number, self.AOI,number=1)
                                    result_AOI_2 = fixation_AOI(eye_tracker_data, video, [middle, end_scene], part_number, scene_number, self.AOI,number=2)

                                    amplitude = self.saccade_amplitude(eye_tracker_data[i_1:i_2])
                                    velocity = self.saccade_velocity(eye_tracker_data[i_1:i_2])
                                    duration = self.duration_fixation(eye_tracker_data[i_1:i_2])
                                    result_participant  = {
                                            'Participant number': part_number,
                                            'Scenario number': scene_number,
                                            'Saccade amplitude': amplitude,
                                            'Saccade velocity': velocity,
                                            'Duration fixation': duration,
                                            'fixation in side mirror': result_AOI[0]["side mirror"] + result_AOI_2[0]["side mirror"],
                                            'fixation in rear mirror': result_AOI[0]["reer mirror"] + result_AOI_2[0]["reer mirror"],
                                            'fixation in speed': result_AOI[0]["speed"] + result_AOI_2[0]["speed"],
                                            'duration fixation in side mirror': result_AOI[1]["side mirror"] + result_AOI_2[1]["side mirror"],
                                            'duration fixation in rear mirror': result_AOI[1]["reer mirror"] + result_AOI_2[1]["reer mirror"],
                                            'duration fixation in speed': result_AOI[1]["speed"] + result_AOI_2[1]["speed"],
                                            'number speed in view': result_AOI[2]["speed"] + result_AOI_2[2]["speed"],
                                            'number rear mirror in view': result_AOI[2]["reer mirror"] + result_AOI_2[2]["reer mirror"],
                                            'number side mirror in view': result_AOI[2]["side mirror"] + result_AOI_2[2]["side mirror"],

                                        }
                                
                                
                                    final_data.append(result_participant)

        df = pd.DataFrame(final_data)

        df.to_csv('participant_result_eye.csv', mode='a', index=False, header=not os.path.isfile('participant_result_eye.csv'))                            
                    