import os
import ijson
import numpy as np
import math

DIR = '../../open-data/data/events'
"""
Formats vectors with following features:
    0: statsbomb_xg
    1: outcome
    2-3: shot coordinates
    4: header (binary)
    5: left foot (binary)
    6: right foot (binary)
    7-8: goalie coordinates
    9-28: defender coordinates sorted by closeness to shot
    29-48: teammate coordinates sorted by closeness to shot
"""

def distance(shot, player):
    return math.sqrt((player[0] - shot[0])**2 + (player[1] - shot[1])**2)

def parse_object(parser):
    _, event, val = next(parser)
    obj = {}
    while event != ('end_map'):
        key = val
        _, event, val = next(parser)
        if event == 'start_map':
            val, parser = parse_object(parser)
        if event == 'start_array':
            val, parser = parse_array(parser)
        obj[key] = val
        _, event, val = next(parser)
    return (obj, parser)

def parse_array(parser):
    _, event, val = next(parser)
    arr = []
    while event != 'end_array':
        if event == 'start_map':
            val, parser = parse_object(parser)
        elif event == 'start_array':
            val, parser = parse_array(parser)
        arr.append(val)
        _, event, val = next(parser)
    return arr, parser

def get_shots_from_file(filename):
    with open(filename, 'rb') as file:
        parser = ijson.parse(file)
        shots = []
        for pre, event, val in parser:
            if event == 'start_array' or event == 'end_array':
                continue
            if (pre, event) == ('item', 'start_map'):
                obj = {}
            elif (pre, event) == ('item', 'end_map'):
                #SELECT FEATURES HERE
                if obj['type']['id'] == 16:
                    shot = [
                        obj['shot']['statsbomb_xg'],                            #StatsBomb xG
                        1 if obj['shot']['outcome']['name'] == 'Goal' else -1,  #Outcome
                        obj['location'][0],                                     #Shot X Coordinate
                        obj['location'][1],                                     #Shot Y Coordinate
                        1 if obj['shot']['body_part']['id'] == 37 else 0,       #Header?
                        1 if obj['shot']['body_part']['id'] == 38 else 0,       #Left Foot?
                        1 if obj['shot']['body_part']['id'] == 40 else 0        #Right Foot?
                    ]
                    #get player positions, ignoring penalty kicks
                    if obj['shot']['type']['id'] != 88:
                        freeze_frame = obj['shot']['freeze_frame']

                        #goalie
                        goalie = [p['location'] for p in freeze_frame if p['teammate'] == False and p['position']['id'] == 1]
                        if not goalie:
                            goalie = [60, 40]
                        else: 
                            goalie = np.array(goalie.flatten())
                        shot.extend(goalie)

                        #defenders sorted by closeness to ball
                        defenders = [p['location'] for p in freeze_frame if p['teammate'] == False and p['position']['id'] != 1]
                        defenders.sort(key=lambda x: distance(obj['location'], x))
                        defenders = np.array(defenders).flatten()
                        shot.extend(defenders)
                        shot.extend([60, 40] * int((20 - len(defenders)) / 2))

                        #teammates sorted by closeness to ball
                        teammates = [p['location'] for p in freeze_frame if p['teammate'] == True ]
                        teammates.sort(key=lambda x: distance(obj['location'], x))
                        teammates = np.array(teammates).flatten()
                        shot.extend(teammates)
                        shot.extend([60, 40] * int((20 - len(teammates)) / 2))

                        shots.append(shot)
                    else:
                        continue
            else:
                key = val
                pre, event, val = next(parser)
                #recursively parse object/array vals
                if event == 'start_map':
                    val, parser = parse_object(parser)
                if event == 'start_array':
                    val, parser = parse_array(parser)
                obj[key] = val
    return np.array(shots)


shots = []
for filename in os.listdir(DIR):
    shots_from_file = get_shots_from_file(DIR + '/' + filename)
    shots.extend(shots_from_file)

print("Successfully loaded " + str(len(shots)) + " into shots.npy")
np.save('shots.npy', shots)