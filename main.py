import math
import os
import shutil
import numpy as np
import pandas as pd
import argparse
import torch
import model
import utils
import crystal_conversion



def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_original_features", type=str, default="./original original_features"
    )
    parser.add_argument("--path_features", type=str, default="./features")
    parser.add_argument("--path_xsd", type=str, default="./original xsd")
    parser.add_argument("--path_cif", type=str, default="./original cif")
    parser.add_argument("--path_npy1", type=str, default="./original npy1")
    parser.add_argument("--path_npy2", type=str, default="./original npy2")
    parser.add_argument("--path_label1", type=str, default="./original label1")
    parser.add_argument("--path_label2", type=str, default="./original label2")
    parser.add_argument("--path_labels", type=str, default="./labels.csv")
    parser.add_argument("--model_save_path", type=str, default="./models")
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--grid_density", type=int, default=1)
    parser.add_argument("--new_shape", type=list, default=[32, 32, 32])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lengths", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--color_channel", type=int, default=7)
    parser.add_argument("--device", type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    
    return parser

parser = get_parse()
args = parser.parse_args(args=[])

#crystal_conversion.xsd_to_cif(args)
#crystal_conversion.cif_to_grid(args)
#crystal_conversion.resize_data(args)
#utils.creatlabel1(args)
#utils.creatlabel2(args)
model.train_loader(args)
model.test_loader(args)
model.train(args)