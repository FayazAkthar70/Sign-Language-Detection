#python script to make folder structure  for project.

import os
import numpy as np

DATA_PATH = os.path.join('Data')
actions = np.array(['Hello', 'Thanks', 'ILoveYou'])
no_sequence = 30
sequence_length = 50
start_folder = 0

if __name__ == '__main__':
    for action in actions:
        for sequence in range(no_sequence):
            try:
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass
    