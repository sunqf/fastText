#!/usr/bin/env python

import sys
import random


dict = {}

with open(sys.argv[1]) as input:
    data = [[item.strip() for item in line.split(" ,", 1)] for line in input]

with open("train.label", "w") as label_file, open("train.first", "w") as first_file, open("train.second", "w") as second_file:
    for first_index, first in enumerate(data):
        second_index = random.randint(0, len(data) - 1)
        flag = second_index % 3;
        while (first_index == second_index or (flag == 0 and first[0] != data[second_index][0])):
            second_index = random.randint(0, len(data) - 1)

        second = data[second_index]

        if first[0] == second[0]:
            label = "1"
        else:
            label = "0"
        label_file.write(label + "\t" + first[1] + "\t" + second[1] + "\n")
        first_file.write(first[1] + "\n")
        second_file.write(second[1] + "\n")
