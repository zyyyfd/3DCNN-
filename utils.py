from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import argparse
import os

def creatlabel1(args):
    path_labels = args.path_labels
    path_label = args.path_label1

    pd_labels = pd.read_csv(path_labels)
    np_labels = np.array(pd_labels)

    try:
        os.makedirs(path_label)
        print("label1 folder has been created successfully")
    except FileExistsError:
        print("label1 folder already exists")
    
    for i in range(0, len(np_labels)):
        label = np_labels[i][0]
        label_name = i + 1
        label_name = int(label_name)
        label_name = path_label + "/" + str(label_name) + ".npy"
        np.save(label_name, label)

def creatlabel2(args):
    path_labels = args.path_labels
    path_label = args.path_label2

    pd_labels = pd.read_csv(path_labels)
    np_labels = np.array(pd_labels)

    try:
        os.makedirs(path_label)
        print("label2 folder has been created successfully")
    except FileExistsError:
        print("label2 folder already exists")
    
    for i in range(0, len(np_labels)):
        label = np_labels[i][2]
        label_name = i + 1
        label_name = int(label_name)
        label_name = path_label + "/" + str(label_name) + ".npy"
        np.save(label_name, label)



