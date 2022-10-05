import csv
import pickle
import numpy as np

with open('../data/mainland_fips_master.csv', encoding='ISO-8859-1') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    list_codes, count = [], 0
    for row in csv_reader:
        if(count > 0 and len(row[0]) == 4):
            code_to_app = "0" + row[0]
            list_codes.append(code_to_app)
        if(count > 0 and len(row[0]) == 5):
            list_codes.append(row[0])
        count += 1

all_codes = dict()
dict_codes = dict()
curr_ct = 0
for val in list_codes:
    dict_codes[val] = 0
    curr_ct += 1

csv_data = []
with open('../data/daily_county2county_2021_04_15_int.csv', encoding='UTF-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    count = 0
    for row in csv_reader:
        if(count > 0):
            csv_data.append(row)
        count += 1

for curr_code in list_codes:
    tot = 0
    for row in csv_data:
        if(row[1] == curr_code):
            new_add = int(row[7]) + int(row[8])
            if(row[0] in dict_codes):
                tot += new_add
                dict_codes[row[0]] = new_add
    if(tot != 0):
        for key in dict_codes:
            dict_codes[key] = dict_codes[key] / tot
    else: dict_codes[curr_code] = 1
    copy_dict = dict_codes.copy()
    all_codes[curr_code] = copy_dict
    for val in list_codes:
        dict_codes[val] = 0

path = '../mobility/mobility_dict.pkl'

with open(path, 'wb') as f:
    pickle.dump(all_codes, f)