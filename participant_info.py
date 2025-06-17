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

import pandas as pd

class ParticipantInfo:
    def __init__(self, info_path):
        self.df = pd.read_csv(info_path)

    def get_age(self, participant_number):
        return self._safe_lookup(participant_number, 'Age')

    def get_gender(self, participant_number):
        return self._safe_lookup(participant_number, 'Gender')

    def get_event_experience(self, participant_number):
        return self._safe_lookup(participant_number, 'How many events have you attended')

    def has_driving_license(self, participant_number):
        return self._safe_lookup(participant_number, 'Do you have a driving license?')

    def _safe_lookup(self, participant_number, column_name):
        row = self.df[self.df['Participant number '] == int(participant_number)]
        if not row.empty and column_name in row.columns:
            return row[column_name].iloc[0]
        else:
            return None
