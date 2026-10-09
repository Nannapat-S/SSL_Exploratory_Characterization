import itertools
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


def IDlookup(alloy_ID, comp_ID, data_root):
    WCO_T = load_WCO_data(data_root, comp_ID)

    list_original = AlloyIDMap_df[AlloyIDMap_df['Alloy ID'] == alloy_ID].iloc[0]
    list_ = list_original.drop('C')

    columns_to_check = ['x', 'y', 'z']
    is_not_zero = [CompIDMap_df.at[comp_ID-1, col] != 0 for col in columns_to_check]
  
    if not is_not_zero[0] and is_not_zero[1] and is_not_zero[2]: # case for A=0 only (i.e., compID1)
        list_ = list_.drop('A')
        wco = np.delete(WCO_T[alloy_ID-1][comp_ID-1], slice(0, 4), axis=0)
    elif is_not_zero[0]  and not is_not_zero[1] and is_not_zero[2]: # case for B=0 only (i.e., compID6)
        list_ = list_.drop('B')
        wco = np.delete(WCO_T[alloy_ID-1][comp_ID-1], slice(4, 7), axis=0)
    elif not is_not_zero[0] and not is_not_zero[1] and is_not_zero[2]: # case for A=0 & B=0 (ternary alloy compID21)
        list_ = list_.drop('A')
        list_ = list_.drop('B')
        wco = np.delete(WCO_T[alloy_ID-1][comp_ID-1], slice(0, 7), axis=0)
    else: # case for quinary alloy 
        wco = WCO_T[alloy_ID-1][comp_ID-1]

    elements = [list_.get(key) for key in ['A', 'B', 'C1', 'C2', 'C3'] if key in list_]
    elements_original = [list_original.get(key) for key in ['A', 'B', 'C'] if key in list_original]
    pairs = [f"{a}-{b}" for a, b in (sorted([x, y]) for x, y in itertools.combinations(elements, 2))]

    pairs_wco_dict = {pair: wco[i] for i, pair in enumerate(pairs)}
    # print(pairs)
    return pairs_wco_dict, elements_original,elements

def compute_WC_ensemble(comp_ID, data_root):
    pair_sum_dict = {}
    pair_count_dict = {}

    # Alloy IDs 9–10 are excluded because their temperature-series
    # data are incomplete.
    complete_alloy_ids = range(1, 9)

    for alloy_ID in complete_alloy_ids:
        pairs_wco_dict, _, _ = IDlookup(
            alloy_ID,
            comp_ID,
            data_root,
        )

        for pair_key, value in pairs_wco_dict.items():
            if pair_key not in pair_sum_dict:
                pair_sum_dict[pair_key] = np.zeros_like(value)
                pair_count_dict[pair_key] = 0

            pair_sum_dict[pair_key] += value
            pair_count_dict[pair_key] += 1

    return {
        pair_key: pair_sum_dict[pair_key] / pair_count_dict[pair_key]
        for pair_key in pair_sum_dict
    }

n_alloyID = 10
alloyID_filename = ['1_NiZn','2_FeZn','3_FeNi','4_CuZn','5_CuNi','6_CuFe','7_CoZn','8_CoNi','9_CoFe','10_CoCu']
n_compID = 21
temp_start = 300
temp_final = 2100
temperature = np.arange(temp_start,temp_final+50, 50)


@lru_cache(maxsize=1)
def load_WCO_data(data_root, comp_ID):
    WCO = np.zeros((n_alloyID,n_compID,len(temperature)),dtype=object)
    WCO_stacked = np.empty((n_alloyID, n_compID), dtype=object)
    wco_line = 69

    # Retrieve WCO from recheck_energy.txt.
    for l in range(n_alloyID):
        for m in [comp_ID - 1]:
            for n,temp in enumerate(temperature):
                if l == 8-1 and m == 16-1 and temp in [1500,2000]:
                    continue # missing data
                if l == 9-1 and m == 1-1 and temp in [550]:
                    continue # missing data
                if l == 9-1 and m == 21-1 and temp in [550]:
                    continue # missing data
                filepath = Path(data_root) / str(l+1) / str(m+1) / str(temp) / "recheck_energy.txt"
                with open(filepath,"r") as myfile:
                    for i, line in enumerate(myfile, start=1):
                        if i == wco_line:
                            wco_values = line.split()
                            wco_values = [float(value) for value in wco_values] # convert to float
                            WCO[l, m, n] = np.array(wco_values)
                            if WCO_stacked[l, m] is None:
                                WCO_stacked[l, m] = WCO[l, m, n]
                            else:
                                WCO_stacked[l, m] = np.vstack((WCO_stacked[l, m], WCO[l, m, n]))

    # Reshape WCO
    WCO_T = np.empty((n_alloyID, n_compID), dtype=object)
    for l in range(n_alloyID):
        for m in [comp_ID - 1]:
            WCO_T[l, m] = np.transpose(WCO_stacked[l, m])
    return WCO_T


# Create alloy name mapping through discrete variables modeling

AlloyIDMap = {
    "Alloy ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "A": ["Ni", "Fe", "Fe", "Cu", "Cu", "Cu", "Co", "Co", "Co", "Co"],
    "B": ["Zn", "Zn", "Ni", "Zn", "Ni", "Fe", "Zn", "Ni", "Fe", "Cu"],
    "C": ["Co-Cu-Fe", "Co-Cu-Ni", "Co-Cu-Zn", "Co-Fe-Ni", "Co-Fe-Zn", "Co-Ni-Zn", "Cu-Fe-Ni", "Cu-Fe-Zn", "Cu-Ni-Zn", "Fe-Ni-Zn"]
}
AlloyIDMap_df = pd.DataFrame(AlloyIDMap)
AlloyIDMap_df[['C1', 'C2', 'C3']] = AlloyIDMap_df['C'].str.split('-', expand=True)

CompIDMap = {
    'Composition ID': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
    'x': [0,10,20,30,40,50,0,10,20,30,40,0,10,20,30,0,10,20,0,10,0],
    'y': [50,40,30,20,10,0,40,30,20,10,0,30,20,10,0,20,10,0,10,0,0],
    'z': [50,50,50,50,50,50,60,60,60,60,60,70,70,70,70,80,80,80,90,90,100]
    }
CompIDMap_df_original = pd.DataFrame(CompIDMap)
CompIDMap_df = pd.DataFrame(CompIDMap)
CompIDMap_df['z1'] = CompIDMap_df['z'] / 3
CompIDMap_df['z2'] = CompIDMap_df['z'] / 3
CompIDMap_df['z3'] = CompIDMap_df['z'] / 3
