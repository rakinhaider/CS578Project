#!/usr/bin/env python
# coding: utf-8

from pymatgen import MPRester
import pandas as pd
from ase.data import atomic_numbers, chemical_symbols

#Read the task ids of single element reference
df_ref = pd.read_excel("Ref_atoms_taskid.xlsx")
ref_ids=df_ref["task_id"].tolist()

#Get property values of single element reference
with MPRester("1ZEWhG2nVio5ykmM") as mpr:
    data_ref = mpr.query(criteria={"task_id": {"$in": ref_ids}}, properties=["task_id","formula","density",
                                                                         "spacegroup.number","band_gap.search_gap.band_gap",
                                                                         "magnetism.total_magnetization_normalized_vol",
                                                                         "e_above_hull","final_energy_per_atom"])

#Transfer property values of single element reference to database
df_ref_data=pd.DataFrame(data_ref)
df_ref_data=df_ref_data.sort_values(by=['task_id'])
df_ref=df_ref.sort_values(by=['task_id'])
dref_final=pd.merge(df_ref, df_ref_data, on="task_id")
#dref_final.to_excel("output_ref_final.xlsx")

#Get task ids of oxides
with MPRester("1ZEWhG2nVio5ykmM") as mpr:   
    entries = mpr.query({"$and":[{"elements": "O"},{"elements":{"$in":chemical_symbols}}] , "nelements": {"$eq": 3}}, ["task_id","pretty_formula"])
oxide_ids = [e['task_id'] for e in entries]

#Get property values of oxides
with MPRester("1ZEWhG2nVio5ykmM") as mpr:
    data = mpr.query(criteria={"task_id": {"$in": oxide_ids}}, properties=["task_id","formula","density","oxide_type",
                                                                         "spacegroup.number","band_gap.search_gap.band_gap",
                                                                         "magnetism.total_magnetization_normalized_vol",
                                                                         "structure.lattice.alpha","structure.lattice.gamma",
                                                                         "structure.lattice.alpha","structure.lattice.beta",
                                                                         "e_above_hull"])

#Transfer to database
df=pd.DataFrame(data)
df_label=df["e_above_hull"]
del df["e_above_hull"]

#Defining functions to be applied on database
###############################################################################

def calc1(row):
    #row["formula"]=(row["formula"])
    row["formula"]=(row["formula"])
    keys=row["formula"].keys()
    O_rat=row["formula"]["O"]
    S=[]
    for items in keys:
        row["formula"][items]=row["formula"][items]/O_rat
        if items != 'O':
            S.append(items)
    if atomic_numbers[S[0]]>atomic_numbers[S[1]]:
        S[0],S[1]=S[1],S[0]
    at_1 = dref_final.loc[dref_final['pretty_formula'] == S[0]]
    at_2 = dref_final.loc[dref_final['pretty_formula'] == S[1]]
    spacegroup_1 = at_1["spacegroup.number"].values[0]
    spacegroup_2 = at_2["spacegroup.number"].values[0]
    mag_1 = at_1["magnetism.total_magnetization_normalized_vol"].values[0]*100
    mag_2 = at_2["magnetism.total_magnetization_normalized_vol"].values[0]*100
    fin_energy_1 = at_1["final_energy_per_atom"].values[0]
    fin_energy_2 = at_2["final_energy_per_atom"].values[0]
    fin_energy=(fin_energy_1+fin_energy_2)/2
    return pd.Series([atomic_numbers[S[0]],atomic_numbers[S[1]],row["formula"][S[0]],row["formula"][S[1]],
                      spacegroup_1,spacegroup_2,mag_1,mag_2,fin_energy],
                     index=["atomic_num_1","atomic_num_2","ratio_1","ratio_2",
                            "spacegroup_1","spacegroup_2","mag_1","mag_2","fin_energy_ref"])


def type_of_oxide(row):
    if row["oxide_type"]=="oxide":
        y=0 
    elif row["oxide_type"]=="peroxide":
        y=1
    elif row["oxide_type"]=="hydroxide":
        y=2
    elif row["oxide_type"]=="superoxide":
        y=3
    elif row["oxide_type"]=="ozonide":
        y=4
    return y

def label(row):
    if row["e_above_hull"]<0.04: #Anything above 40 meV is considered unstable
        y=1 #stable
    else:
        y=0 #unstable
    return y

###############################################################################

#Applying fubctions on X_data
df_a = df.apply(calc1, axis = 1)
df = pd.concat([df,df_a], axis=1)
df["oxide_type"] = df.apply(type_of_oxide, axis = 1)
df["magnetism.total_magnetization_normalized_vol"]=df["magnetism.total_magnetization_normalized_vol"]*100
df.to_excel("X_data.xlsx")

#Applying fubctions on Y_label
df_label=df_label.to_frame()
df_label = df_label.apply(label, axis = 1)
df_label.to_excel("y_label.xlsx")





