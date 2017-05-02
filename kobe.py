"""
K     K     O O O     B B B     E E E E E
K   K     O       O   B     B   E
K K       O       O   B B B     E E E E E
K K       O       O   B     B   E
K   K     O       O   B     B   E
K     K     O O O     B B B     E E E E E


For preprocessing, get rid of data points that don't have shot made field.
Create test/training splits from data that does have shot made field.
"""
import csv
from sklearn.ensemble import *
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.cross_validation import KFold
def randomForestStrToNum():
    # turn categorical variables into dummy variables
    categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'period', 'season']
    for var in categorical_vars:
        raw = pd.concat([raw, pd.get_dummies(raw[var], prefix=var)], 1)
        raw = raw.drop(var, 1)
def main():
    # import data; maybe make new function for preprocessing data
    filename= "data.csv"
    raw = pd.read_csv(filename)
    nona =  raw[pd.notnull(raw['shot_made_flag'])]
    nona['remaining_time'] = ['minutes_remaining'] * 60 + nona['seconds_remaining']
    nona["last_5_sec_in_period"] = nona["remaining_time"] < 5
    drops = ["minutes_remaining", "seconds_remaining","team_id", "shot_zone_area", 'shot_zone_range', 'shot_zone_basic', "game_date", "team_name", "matchup", "lat", "lon", 'game_event_id']
    nona["home_play"] = nona["matchup"].str.contains("vs").astype("int")
    for drop in drops:
        nona = nona.drop(drop, 1)
    train = nona.drop('shot_made_flag', 1)
    train_y = nona['shot_made_flag']
    
main()
