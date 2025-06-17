from ultralytics import YOLO
import cv2
import pandas as pd
from collections import Counter
from collections import defaultdict
from trim_video import trim_video
import os

def save_frame_from_video(video_path, output_folder, timestamp, participant_nb, scene_nb):
    """
    Saves a frame from the video at a specific timestamp.

    Parameters:
    - video_path: Path to the video file.
    - output_folder: Folder to save the extracted frame.
    - timestamp: Time in seconds from the start to extract the frame.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Set the frame position at the desired timestamp
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    frame_position = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)

    # Read the frame
    ret, frame = cap.read()
    if ret:
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Create the output file path
        output_path = os.path.join(output_folder, f"frame_at_{timestamp}s_{participant_nb}_{scene_nb}.jpg")

        # Save the frame as an image
        cv2.imwrite(output_path, frame)
        print(f"Frame saved at: {output_path}")
    else:
        print("Error: Could not read the frame.")

    # Release the video capture object
    cap.release()

def fixation_AOI(eye_data, video_path, time_partition, part_number, scene_nb, dict_AOI,number = None):
                 
    check = eye_data
    trim_video(video_path, time_partition[0], time_partition[1])
    out_video_path = "trimmed_video.mp4"
    save_frame_from_video("trimmed_video.mp4", "frame_check", 5, part_number, scene_nb)


    fixation_durations = check.groupby('fixation_number').agg(
        start_time=('timestamp', 'first'),
        end_time=('timestamp', 'last')
    )

    fixation_durations['duration'] = fixation_durations['end_time'] - fixation_durations['start_time']

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) 
    cap.release()

    timestep = 1 / fps

    model = YOLO("runs/detect/train2/weights/best.pt")

    results = model(out_video_path, show=False, save=False)

    count = 0

    #number_view = {"side mirror": 0, "reer mirror": 0, "speed": 0}
    number_view = {}
    eye_dict = {}
    #eye_dict = {"side mirror": 0, "reer mirror": 0, "speed": 0}
    #conversion = {0: "speed", 1: "reer mirror", 2: "side mirror"}
    for AOI_of_choice in dict_AOI:
        eye_dict[dict_AOI[AOI_of_choice]] = 0
        number_view[dict_AOI[AOI_of_choice]] = 0


    time = 0
    current_fix = 0
    fix_list = []
    durations = []
    result = {
        "time": [],
        "class": [],
        "AOI": [],
        "xmax": [],
        "xmin": [],
        "ymax": [],
        "ymin": [],

    }

    for frame_idx, frame_results in enumerate(results):
        timestamp = (frame_idx * timestep) + time_partition[0]
        

        for i in range(time,len(check)):
                
                t_eye = check.iloc[i]["timestamp"] 
                if (timestamp - t_eye) <= 0:
                    
                    time = i
                    break
        

       
        fix = False
        p = 0
        
        for box in frame_results.boxes:
            
            x_min, y_min, x_max, y_max = box.xyxy[0]  
            cls = box.cls[0]  
            for AOI_class in range(len(dict_AOI)):
                if int(cls.item()) == AOI_class:
                    number_view[dict_AOI[AOI_class]] += 1

            #if int(cls.item()) == 0:
            #    number_view[dict_AOI[0]] += 1
                
        
            #if int(cls.item()) == 1:
            #    number_view[dict_AOI[1]] += 1 
                
               
            #if int(cls.item()) == 2:
            #    number_view[dict_AOI[2]] += 1
                
                
            result["time"].append(timestamp)
            result["class"].append(int(cls.item()))
            result["AOI"].append(dict_AOI[int(cls.item())])
            result["xmax"].append(x_max)
            result["xmin"].append(x_min)
            result["ymax"].append(y_max)
            result["ymin"].append(y_min)

            

          
        
            if check.iloc[time]["eye_movement"] == "fixation":
                if int(check.iloc[time]["fixation_number"]) == current_fix:
                    fix = True

                
                if float(check.iloc[time]["gaze2d_x"]) > (x_min - 10) and float(check.iloc[time]["gaze2d_x"]) < (x_max + 10):
                    if check.iloc[time]["gaze2d_y"] > (y_min - 10) and check.iloc[time]["gaze2d_y"] < (y_max + 10):
                        count += 1
                        if int(check.iloc[time]["fixation_number"]) != current_fix:
                            
                            print(fix_list)
                            if current_fix != 0:
                                duration = fixation_durations.loc[int(current_fix), 'duration']
                                
                            current_fix = int(check.iloc[time]["fixation_number"])
                            #if count > 4:
                            #   breakpoint()
                            
                            try:
                                print(Counter(fix_list).most_common(1)[0][0])
                                eye_dict[Counter(fix_list).most_common(1)[0][0]] += 1
                                durations.append((Counter(fix_list).most_common(1)[0][0], duration))
                            except:
                                pass
                        
                            fix_list = []
                            
                            for AOI_class in range(len(dict_AOI)):
                                if int(cls.item()) == AOI_class:
                                    fix_list.append(dict_AOI[AOI_class])
                                    p+=1
                                    
                            
                            #if int(cls.item()) == 0:
                            #    fix_list.append("speed")
                            #    p += 1
                            

                            #if int(cls.item()) == 1:
                            #    fix_list.append("reer mirror")
                            #    p += 1
                            #if int(cls.item()) == 2:
                            #    fix_list.append("side mirror")
                            #    p += 1
                            
                            
                        else:
                            for AOI_class in range(len(dict_AOI)):
                                if int(cls.item()) == AOI_class:
                                    fix_list.append(dict_AOI[AOI_class])
                                    p+=1
        if p == 0 and fix == True:
        
            fix_list.append("None")

                    
    



    # Dictionary to store sum and count per class
    class_sums = defaultdict(float)
    class_counts = defaultdict(float)

    # Aggregate sums and counts
    for cls, value in durations:
        class_sums[cls] += value
        class_counts[cls] += 1

    # Calculate averages
    class_averages = {cls: class_sums[cls] / class_counts[cls] for cls in class_sums}
    
    required_classes = list(dict_AOI.values())

    for cls in required_classes:
        if cls not in class_averages:
            class_averages[cls] = 0.0
    df = pd.DataFrame(result)

    # Save DataFrame to CSV
    if number == 2:
        filename = f"{part_number}_{scene_nb}_{number}_fixation_AOI.csv"
    elif number == 1:
        filename = f"{part_number}_{scene_nb}_{number}_fixation_AOI.csv"
    else:
        filename = f"{part_number}_{scene_nb}_fixation_AOI.csv"
    df.to_csv(f"fixation_results/{filename}", index=False)
    
    return (eye_dict, class_averages, number_view)
    


eye_data = pd.read_csv("combined_data.csv")
video_path = "normal_analysis.mp4"

#fixation_AOI(eye_data, video_path)