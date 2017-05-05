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
# from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.cross_validation import KFold, cross_val_score



def randomForestStrToNum(raw):
    # turn categorical variables into dummy variables
    categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'period', 'season']
    for var in categorical_vars:
        dvVar = pd.get_dummies(raw[var])
        raw = pd.concat([raw, dvVar], axis=1)
        raw = raw.drop(var, 1)
    return raw

def testModel(model, train, train_y, num_rounds, folds):
    # performs cross_validation on kfolds, returns the average over all rounds

    avg_total = 0
    for i in range(num_rounds):
        results = cross_val_score(model, train, train_y, cv=folds)
        avg_round = sum(results) / 3
        print("Results: %s, Average: %f" % (results, avg_round))
        avg_total += avg_round

    return avg_total / num_rounds

def main():
    # import data; maybe make new function for preprocessing data
    filename= "data.csv"
    raw = pd.read_csv(filename)
    raw['remaining_time'] = raw['minutes_remaining'] * 60 + raw['seconds_remaining']
    raw["last_5_sec_in_period"] = raw["remaining_time"] < 5
    drops = ["minutes_remaining", "seconds_remaining","team_id", "shot_zone_area", \
             'shot_zone_range', 'shot_zone_basic', "game_date", "team_name", "matchup", "lat", "lon", 'game_event_id']
    raw["home_play"] = raw["matchup"].str.contains("vs").astype("int")
    for drop in drops:
        raw = raw.drop(drop, 1)
    raw = randomForestStrToNum(raw)
    nona =  raw[pd.notnull(raw['shot_made_flag'])]

    #splitting explantory and response variables
    train = nona.drop('shot_made_flag', 1)
    train_y = nona['shot_made_flag']

    #setting up KFolds
    seed = 24
    num_folds = 3
    folds = KFold(len(train), n_folds=num_folds, random_state=seed, shuffle=True)


    model = RandomForestClassifier(n_estimators=150, max_depth = 10)
    model = model.fit(train, train_y)

    num_rounds = 10
    randomForestScore = testModel(model, train, train_y, num_rounds, folds)
    print("Average after %d rounds: %f" % (num_rounds, randomForestScore))

    #test out different parameters for random forest
    #shuffle when running cross_val_score
    #test decision trees
    #SVMS
    #adaboost


main()
