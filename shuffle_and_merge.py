import pandas as pd
import numpy as np
import os
import json
import pathlib
import random

home_folder=pathlib.Path(__file__).parent.resolve()
input_folder=os.path.join(home_folder,"clusters")
output_file=os.path.join(home_folder,"data\\merged_data.tsv")


for filename in os.listdir(input_folder):
    f = os.path.join(input_folder, filename)
    # checking if it is a file
    if os.path.isfile(f):
        lines = open(f).readlines()
        random.shuffle(lines)
        open(f, 'w').writelines(lines)

with open(output_file, 'w') as outfile:
    for filename in os.listdir(input_folder):
        f = os.path.join(input_folder, filename)
        if os.path.isfile(f):
            with open(f) as infile:
                for line in infile:
                    outfile.write(line)

