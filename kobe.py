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
from sklearn.cross_validation import KFold, cross_val_score

##############################################################

#test prediction on specific points (dunks, layups, 3s maybe?)
#look at probabilistic predictions

##############################################################

def randomForestStrToNum(raw):
    # turn categorical variables into dummy variables
    categorical_vars = ['action_type', 'combined_shot_type', 'shot_type', 'opponent', 'period', 'season']
    for var in categorical_vars:
        dvVar = pd.get_dummies(raw[var])
        raw = pd.concat([raw, dvVar], axis=1)
        raw = raw.drop(var, 1)
        # print(raw[var])
        # print(var, dvVar)
    # print(list(raw.columns.values))
    return raw

def testModel(model, train, train_y, num_rounds, folds):
    # performs cross_validation on kfolds, returns the average over all rounds

    avg_total = 0
    for i in range(num_rounds):
        # model.fit(train, train_y)
        results = cross_val_score(model, train, train_y, cv=folds)
        avg_round = sum(results) / 3
        print("Results: %s, Average: %f" % (results, avg_round))
        avg_total += avg_round

    return avg_total / num_rounds


def testSubset(model, train, train_y, sub_train, sub_train_y, num_rounds, num_folds, seed):
    """
    fill
    """
    avg_total = 0
    for train_k, test_k in KFold(len(sub_train), n_folds=num_folds, random_state=seed, shuffle=True):
        model.fit(train, train_y)
        # pred = model.predict(dunk_train.iloc[test_k])
        result = model.score(sub_train.iloc[train_k], sub_train_y.iloc[train_k])
        avg_total += result

    return avg_total / 3

def main():
    # import data; maybe make new function for preprocessing data
    filename= "data.csv"
    raw = pd.read_csv(filename)
    originalFrame = raw.copy()
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

    # setting up KFolds
    seed = 24
    num_folds = 3
    num_rounds = 10

    folds = KFold(len(train), n_folds=num_folds, random_state=seed, shuffle=True)


    model = RandomForestClassifier(n_estimators=200, max_depth =10, max_features=0.25, random_state=seed)
    # model = model.fit(train, train_y)

    #################################################################
    #Looking at specific shots and their predicted probability
    layupFrame = nona.loc[nona["Layup"] == 1]
    fadeawayShotFrame = nona.loc[nona["Fadeaway Jump Shot"] == 1]
    dunkFrame = nona.loc[nona["Dunk"] == 1]


    ################# LAYUPS #########################
    layup_train = train.loc[train["Layup"] == 1]
    layup_train_y = layupFrame['shot_made_flag']

    print("LayupFrame shape: " + str(layupFrame.shape))
    layupScore = testSubset(model, train, train_y, layup_train, layup_train_y, num_rounds, num_folds, seed)
    print(layupScore)

    ################## DUNKS ########################
    dunk_train = train.loc[train["Dunk"] == 1]
    dunk_train_y = dunkFrame['shot_made_flag']

    print("DunkFrame shape: " + str(dunkFrame.shape))
    dunkScore = testSubset(model, train, train_y, dunk_train, dunk_train_y, num_rounds, num_folds, seed)
    print(dunkScore)

    ################## FADEAWAY SHOTS ########################
    # fade_train = train.loc[train["Fadeaway Jump Shot"] == 1]
    # fade_train_y = fadeawayShotFrame['shot_made_flag']
    #
    # print("FadeawayShotFrame shape: " + str(fadeawayShotFrame.shape))
    # fadeScore = testSubset(model, train, train_y, fade_train, fade_train_y, num_rounds, num_folds, seed)
    # print(fadeScore)





main()
