import pandas as pd
import json
import gzip

def convert(imu, gaze):
    
    gaze_data = gaze
    imu_data = imu

    # Function to load line-delimited JSON
    def load_line_delimited_json(file_path):
        data = []
        with gzip.open(file_path, 'rt', encoding='utf-8') as file:  # 'rt' mode for reading text
            for line in file:
                # Skip empty lines or whitespace
                if line.strip():
                    data.append(json.loads(line.strip()))
        
        return data

    # Load the gaze data and IMU data
    gaze_json_data = load_line_delimited_json(gaze_data)
    imu_json_data = load_line_delimited_json(imu_data)

    # Flatten JSON data into a tabular format for a DataFrame
    def flatten_gaze_data(json_data):
        rows = []
        for entry in json_data:
            row = {
                "timestamp": entry.get("timestamp"),
            }
            data = entry.get("data", {})
            row.update({
                "gaze2d_x": data.get("gaze2d", [None, None])[0],
                "gaze2d_y": data.get("gaze2d", [None, None])[1],
                "gaze3d_x": data.get("gaze3d", [None, None, None])[0],
                "gaze3d_y": data.get("gaze3d", [None, None, None])[1],
                "gaze3d_z": data.get("gaze3d", [None, None, None])[2],
            })
            eyeleft = data.get("eyeleft", {})
            row.update({
                "eyeleft_gazeorigin_x": eyeleft.get("gazeorigin", [None, None, None])[0],
                "eyeleft_gazeorigin_y": eyeleft.get("gazeorigin", [None, None, None])[1],
                "eyeleft_gazeorigin_z": eyeleft.get("gazeorigin", [None, None, None])[2],
                "eyeleft_gazedirection_x": eyeleft.get("gazedirection", [None, None, None])[0],
                "eyeleft_gazedirection_y": eyeleft.get("gazedirection", [None, None, None])[1],
                "eyeleft_gazedirection_z": eyeleft.get("gazedirection", [None, None, None])[2],
                "eyeleft_pupildiameter": eyeleft.get("pupildiameter"),
            })
            eyeright = data.get("eyeright", {})
            row.update({
                "eyeright_gazeorigin_x": eyeright.get("gazeorigin", [None, None, None])[0],
                "eyeright_gazeorigin_y": eyeright.get("gazeorigin", [None, None, None])[1],
                "eyeright_gazeorigin_z": eyeright.get("gazeorigin", [None, None, None])[2],
                "eyeright_gazedirection_x": eyeright.get("gazedirection", [None, None, None])[0],
                "eyeright_gazedirection_y": eyeright.get("gazedirection", [None, None, None])[1],
                "eyeright_gazedirection_z": eyeright.get("gazedirection", [None, None, None])[2],
                "eyeright_pupildiameter": eyeright.get("pupildiameter"),
            })
            rows.append(row)
        return rows

    def flatten_imu_data(json_data):
        rows = []
        for entry in json_data:
            row = {
                "timestamp": entry.get("timestamp"),
            }
            data = entry.get("data", {})
            row.update({
                "accelerometer_x": data.get("accelerometer", [None, None, None])[0],
                "accelerometer_y": data.get("accelerometer", [None, None, None])[1],
                "accelerometer_z": data.get("accelerometer", [None, None, None])[2],
                "gyroscope_x": data.get("gyroscope", [None, None, None])[0],
                "gyroscope_y": data.get("gyroscope", [None, None, None])[1],
                "gyroscope_z": data.get("gyroscope", [None, None, None])[2],
            })
            rows.append(row)
        return rows

    # Flatten the data
    flattened_gaze_data = flatten_gaze_data(gaze_json_data)
    flattened_imu_data = flatten_imu_data(imu_json_data)

    # Convert to DataFrames
    gaze_df = pd.DataFrame(flattened_gaze_data)
    imu_df = pd.DataFrame(flattened_imu_data)
    gaze_df['gaze2d_x'] = gaze_df['gaze2d_x'].apply(lambda x: x * 1920 if pd.notnull(x) else x)
    gaze_df['gaze2d_y'] = gaze_df['gaze2d_y'].apply(lambda y: y * 1080 if pd.notnull(y) else y)

    import numpy as np

    # Function to calculate Euclidean distance
    def calculate_distance(x1, y1, x2, y2):
        if pd.notnull(x1) and pd.notnull(y1) and pd.notnull(x2) and pd.notnull(y2):
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return None

    # Add an 'eye_movement' column
    gaze_df['eye_movement'] = 'fixation'  # Default to fixation

    # Calculate distances and classify eye movement
    for i in range(1, len(gaze_df)):
        prev_row = gaze_df.iloc[i - 1]
        current_row = gaze_df.iloc[i]
        
        distance = calculate_distance(
            prev_row['gaze2d_x'], prev_row['gaze2d_y'],
            current_row['gaze2d_x'], current_row['gaze2d_y']
        )
        
        if distance is not None and distance > 40:
            gaze_df.at[i, 'eye_movement'] = 'saccade'

    # Function to correct saccade pairs surrounded by fixations
    def correct_saccade_pairs(df):
        for i in range(2, len(df) - 3):
            if (
                
                df.at[i - 1, 'eye_movement'] == 'fixation' and
                df.at[i, 'eye_movement'] == 'saccade' and
                df.at[i + 1, 'eye_movement'] == 'saccade' and
                df.at[i + 2, 'eye_movement'] == 'fixation' 
            ):
                # Correct the two saccades to fixations
                df.at[i, 'eye_movement'] = 'fixation'
                df.at[i + 1, 'eye_movement'] = 'fixation'

    # Apply the correction
    #correct_saccade_pairs(gaze_df)




    # Add a 'label' column based on the source of the data
    gaze_df['label'] = 'eye tracker'
    imu_df['label'] = ''

    # Combine and sort the data by timestamp
    combined_df = pd.concat([gaze_df, imu_df]).sort_values(by="timestamp").reset_index(drop=True)

    # Reorder columns to make 'label' the second column
    columns = [combined_df.columns[0]] + ['label'] + [col for col in combined_df.columns if col not in ['label', combined_df.columns[0]]]
    combined_df = combined_df[columns]

    # Add a 'fixation_number' column
    fixation_number = 0
    fixation_numbers = []

    previous_movement = None

    for movement in combined_df['eye_movement']:
        if pd.notnull(movement):  # Ignore empty cells
            if movement != previous_movement:
                fixation_number += 1
            previous_movement = movement
        fixation_numbers.append(int(fixation_number) if pd.notnull(movement) else None)

    combined_df['fixation_number'] = fixation_numbers


    # Save to a CSV file
    output_csv_path = "combined_data.csv"
    
    combined_df.to_csv(output_csv_path, index=False)

    return combined_df

convert(
   "/Volumes/Elements/U17CC/ETG_Loughborough/20250219T091649Z/imudata.gz",  "/Volumes/Elements/U17CC/ETG_Loughborough/20250219T091649Z/gazedata.gz"
)
