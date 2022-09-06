# This is to get the renormalized mobility dictionaries for custom FIPS codes
# Note: You might also have to re-run gen_all_mobility_mat.py for the code to successfully read in the .pkl file

import pickle
from math import isclose

with open('mobility/mobility_dict.pkl', 'rb') as f:
    list_mob = pickle.load(f)

def get_mob_custom(list_fips):
    trunc_mob = dict()
    for val in list_fips:
        total_inner = 0
        for v in list_mob[val]:
            total_inner += list_mob[val][v]
        inner_dict = dict()
        for v in list_mob[val]:
            inner_dict[v] = list_mob[val][v] / total_inner
        trunc_mob[val] = inner_dict
    return trunc_mob

# fips_codes = []         # Fill in FIPS codes you want a renormalized mobility dictionary for
fips_codes = ["06037", "17031", "48201", "04013", "06073", "06059", "12086", "48113", "06065", "06071", "32003", "53033", 
            "48439", "26163", "06085", "12011", "48029", "42101", "06001", "25017", "36103", "06067", "36059", "12099", 
            "39035", "12057", "42003", "26125", "39049", "27053", "12095", "51059", "06013",  "49035", "48453", "29189", 
            "04019", "24031", "15003", "36119", "55079", "06019", "47157", "13121", "37119", "17043", "17043"]

mob_custom = get_mob_custom(fips_codes)

with open('mobility/mobility_dict_norm.pkl', 'wb') as f:
    pickle.dump(mob_custom, f)