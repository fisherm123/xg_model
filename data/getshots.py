import os
import ijson
import numpy as np

DIR = '../../open-data/data/events'
"""
Formats feature vectors with following features:
    0: statsbomb_xg
    1: outcome
    2-3: shot coordinates
    4: body part
    5: technique
    6: type
    7-49: player coordinates
"""

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
                #shot type has id 16
                if obj['type']['id'] == 16:
                    length = 7
                    shot = [
                        obj['shot']['statsbomb_xg'],
                        1 if obj['shot']['outcome']['name'] == 'Goal' else 0,
                        obj['location'][0],
                        obj['location'][1],
                        obj['shot']['body_part']['id'],
                        obj['shot']['technique']['id'],
                        obj['shot']['type']['id']
                    ]
                    #if penalty, object has no freeze_frame
                    if 'freeze_frame' in obj['shot']:
                        for p in obj['shot']['freeze_frame']:
                            if p['teammate'] == False:
                                shot.extend(p['location'])
                                length += 2
                        #pad defender locations
                        while length < 29:
                            shot.extend([-1, -1])
                            length += 2
                        for p in obj['shot']['freeze_frame']:
                            if p['teammate'] == True:
                                shot.extend(p['location'])
                                length += 2
                        #pad teammate locations
                        while length < 49:
                            shot.extend([-1, -1])
                            length += 2
                    #if penalty, pad all player location features
                    else:
                        while length < 49:
                            shot.extend([-1, -1])
                            length += 2
                    shots.append(shot)
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