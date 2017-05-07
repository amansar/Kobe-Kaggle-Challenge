"""
T T T T T  E E E E E     S S S    T T T T T     S S S
    T      E           S              T       S
    T      E E E E E     S S S        T         S S S
    T      E                   S      T               S
    T      E E E E E     S S S        T         S S S


Running tests to determine best parameters
"""
import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import *
from sklearn.tree import DecisionTreeClassifier
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

    ###########################################################################

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                         Random Forest Classifier                        #
    #          parameters to test: n_estimators, max_depth, max_features      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # num_estimators = [1, 10, 100, 200]
    # for num in num_estimators:
    #     model = RandomForestClassifier(n_estimators=num, max_depth=10, random_state=seed)
    #     randomForestScore = testModel(model, train, train_y, num_rounds, folds)
    #     print("Number of estimators: %d, Average Score: %f" % (num, randomForestScore))


    # depths = [1, 10, 100, 200]
    # for depth in depths:
    #     model = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=seed)
    #     randomForestScore = testModel(model, train, train_y, num_rounds, folds)
    #     print("Max Depth: %d, Average Score: %f" % (depth, randomForestScore))

    # features = [0.25, 0.5, 0.75, 1.00]
    # for feature in features:
    #     model = RandomForestClassifier(n_estimators=100, max_depth=10, max_features=feature, random_state=seed)
    #     randomForestScore = testModel(model, train, train_y, num_rounds, folds)
    #     print("Max features (percent): %f, Average Score: %f" % (feature, randomForestScore))

    ############################################################################

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                         AdaBoost Classifier                             #
    #          parameters to test: n_estimators, learning_rate                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # num_estimators = [10, 25, 50, 100]
    # for num in num_estimators:
    #     model = AdaBoostClassifier(n_estimators=num, random_state=seed)
    #     adaBoostScore = testModel(model, train, train_y, num_rounds, folds)
    #     print("Number of estimators: %d, Average Score: %f" % (num, adaBoostScore))
    #
    # learning_rates = [0.01, 0.1, 0.5, 1.0]
    # for rate in learning_rates:
    #     model = AdaBoostClassifier(n_estimators=25, learning_rate=rate, random_state=seed)
    #     adaBoostScore = testModel(model, train, train_y, num_rounds, folds)
    #     print("Learning rate: %f, Average Score: %f" % (rate, adaBoostScore))


    ############################################################################

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                         Decision Tree Classifier                        #
    #          parameters to test: max_features, max_depth                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # features = [0.25, 0.5, 0.75, 1.00]
    # for feature in features:
    #     model = DecisionTreeClassifier(max_depth=10, max_features=feature, random_state=seed)
    #     decisionTreeScore = testModel(model, train, train_y, num_rounds, folds)
    #     print("Max features (percent): %f, Average Score: %f" % (feature, decisionTreeScore))

    # depths = [1,5, 10, 25, 50, 75]
    # for depth in depths:
    #     model = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    #     decisionTreeScore = testModel(model, train, train_y, num_rounds, folds)
    #     print("Max depth: %s, Average Score: %f" % (depth, decisionTreeScore))



    ############################################################################

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                         Decision Tree Classifier                        #
    #          parameters to test: max_features, max_depth                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    randomForest = RandomForestClassifier(n_estimators=200, max_depth=10, max_features=0.25, random_state=seed)
    adaBoost = AdaBoostClassifier(n_estimators=25, learning_rate=1.00, random_state=seed)
    decisionTree = DecisionTreeClassifier(max_features=0.25, max_depth=5, random_state=seed)

    classifiers = [randomForest, adaBoost, decisionTree]
    scores = []
    for classifier in classifiers:
        score = testModel(classifier, train, train_y, num_rounds, folds)
        scores.append(score)

    print(scores)


    # randomForestScore = testModel(model, train, train_y, num_rounds, folds)
    # print("Average after %d rounds: %f" % (num_rounds, randomForestScore))

    #test out different parameters for random forest
    #shuffle when running cross_val_score
    #test decision trees
    #SVMS
    #adaboost


main()
