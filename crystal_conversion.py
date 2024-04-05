import math
import numpy as np
import pandas as pd
from ase.io import cif, xsd
from pymatgen.core import Structure
from scipy.ndimage import zoom
import torch
import os
import argparse
import time
import sys
import warnings

def xsd_to_cif(args):
    path_xsd = args.path_xsd
    path_cif = args.path_cif
    try:
        os.makedirs(path_cif)
        print("path_cif folder has been created successfully")
    except FileExistsError:
        print("path_cif folder already exists")
        
    file_list = os.listdir(path_xsd)

    max_count = len(file_list)
    
    for i in range(0, len(file_list)):
        file_path = file_list[i]
        file_path_xsd = path_xsd + "/" + file_path
        atoms = xsd.read_xsd(file_path_xsd)
        file_path_cif = path_cif + "/" + file_path[:-4] + ".cif"
        cif.write_cif(file_path_cif, atoms)
        sys.stdout.write('\r' + f'xsd_to_cif Progress: {i+1}/{max_count}')
    sys.stdout.flush()
    print(" completion of xsd_to_cif task")


def cif_to_grid(args):
    
    warnings.filterwarnings("ignore", message="Issues encountered while parsing CIF: Some fractional coordinates rounded to ideal values to avoid issues with finite precision.")
    
    path_cif = args.path_cif
    path_npy1 = args.path_npy1
    grid_density = args.grid_density

    file_list = os.listdir(path_cif)

    try:
        os.makedirs(path_npy1)
        print("path_npy1 folder has been created successfully")
    except FileExistsError:
        print("path_npy1 folder already exists")

    max_count = len(file_list)
    for y in range(0, len(file_list)):
        file_path = file_list[y]
        file_path_cif = path_cif + "/" + file_path

        structure = Structure.from_file(file_path_cif)

        coords = structure.cart_coords
        coords = coords.astype(np.float32)
        symbols = [str(site.specie) for site in structure.sites]

        symbol_to_atomic_number = {
            "H": 1,
            "C": 2,
            "N": 3,
            "O": 4,
            "S": 5,
            "F": 6,
            "Cl": 7,
            "Br": 8,
            "I": 9,
            "Sn": 10,
            "Ge": 11,
            "Pb": 12,
        }
        symbols_nambers = [symbol_to_atomic_number[symbol] for symbol in symbols]

        symbols_nambers = torch.LongTensor(symbols_nambers)
        symbols_nambers = torch.nn.functional.one_hot(symbols_nambers, num_classes=13)
        symbols_nambers = symbols_nambers.numpy()

        def rounding_decimal1(namber):
            if namber > 0:
                ints1 = int(namber) + 1
            if namber < 0:
                ints1 = int(namber)
            return ints1

        def rounding_decimal2(namber):
            global ints2
            if namber > 0:
                ints2 = int(namber)
            if namber < 0:
                ints2 = int(namber) - 1
            return ints2

        x_list = []
        for i in range(0, len(coords)):
            x_list.append(coords[i][0])
        x_max = max(x_list)
        x_min = min(x_list)
        x_max_int = rounding_decimal1(x_max)
        x_min_int = rounding_decimal2(x_min)

        y_list = []
        for i in range(0, len(coords)):
            y_list.append(coords[i][1])
        y_max = max(y_list)
        y_min = min(y_list)
        y_max_int = rounding_decimal1(y_max)
        y_min_int = rounding_decimal2(y_min)

        z_list = []
        for i in range(0, len(coords)):
            z_list.append(coords[i][2])
        z_max = max(z_list)
        z_min = min(z_list)
        z_max_int = rounding_decimal1(z_max)
        z_min_int = rounding_decimal2(z_min)

        x_namber = (x_max_int - x_min_int) * grid_density
        y_namber = (y_max_int - y_min_int) * grid_density
        z_namber = (z_max_int - z_min_int) * grid_density
        x_original = np.linspace(x_max_int, x_min_int, x_namber)
        y_original = np.linspace(y_max_int, y_min_int, y_namber)
        z_original = np.linspace(z_max_int, z_min_int, z_namber)
        grid_coords_original = np.array([])
        grid_coords_original = np.append(grid_coords_original, x_original)
        grid_coords_original = np.tile(grid_coords_original, y_namber * z_namber)
        y_original1 = np.repeat(y_original, x_namber)
        y_original1 = np.tile(y_original1, z_namber)
        z_original1 = np.repeat(z_original, x_namber * y_namber)
        grid_coords_original = np.hstack(
            (grid_coords_original.reshape(-1, 1), y_original1.reshape(-1, 1))
        )
        grid_coords_original = np.hstack(
            (grid_coords_original, z_original1.reshape(-1, 1))
        )
        grid_coords_original = grid_coords_original.astype(np.float32)

        grid_symbols_original = np.zeros(
            (x_namber, y_namber, z_namber, 13), dtype=np.int8
        )
        grid_tier = np.array([0, 0, 0])
        for i in range(0, len(coords)):
            density1 = 1 / grid_density
            x_tier = int((coords[i][0] - x_min_int - (density1 / 2)) / density1)
            y_tier = int((coords[i][1] - y_min_int - (density1 / 2)) / density1)
            z_tier = int((coords[i][2] - z_min_int - (density1 / 2)) / density1)
            tier_original0 = list([x_tier, y_tier, z_tier])
            grid_tier = np.vstack([grid_tier, np.array(tier_original0)])
        grid_tier = grid_tier[1:]

        for i in range(0, len(coords)):
            x_tier_original = grid_tier[i][0]
            y_tier_original = grid_tier[i][1]
            z_tier_original = grid_tier[i][2]
            grid_symbols_original[x_tier_original][y_tier_original][
                z_tier_original
            ] += symbols_nambers[i]

        file_path_npy = path_npy1 + "/" + file_path[:-4] + ".npy"
        np.save(file_path_npy, grid_symbols_original)
        
        sys.stdout.write('\r' + f'cif_to_grid Progress: {y+1}/{max_count}')
    sys.stdout.flush()
    print(" completion of cif_to_grid task")
    
def resize_data(args):
    import threading

    path_npy1 = args.path_npy1
    path_npy2 = args.path_npy2
    new_shape = args.new_shape

    def zoom_subarray(subarray, zoom_factors, result, index):
        zoomed_subarray = zoom(subarray, zoom_factors)
        result[index] = zoomed_subarray

    num_threads = 13
    file_list = os.listdir(path_npy1)
    #file_list.remove(".ipynb_checkpoints")

    try:
        os.makedirs(path_npy2)
        print("path_npy2 folder has been created successfully")
    except FileExistsError:
        print("path_npy2 folder already exists")
        
    max_count = len(file_list)
    for i in range(0, len(file_list)):
        npy1_path = path_npy1 + "/" + file_list[i]
        
        data = np.load(npy1_path)
        original_shape = data.shape
       
        zoom_factors = (
            new_shape[0] / original_shape[0],
            new_shape[1] / original_shape[1],
            new_shape[2] / original_shape[2],
            1,
        )
       
        subarrays = np.array_split(data, num_threads, axis=3)

        result = [None] * num_threads

        threads = []
        for y in range(0, num_threads):
            thread = threading.Thread(
                target=zoom_subarray, args=(subarrays[y], zoom_factors, result, y)
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        zoomed_data = np.concatenate(result, axis=3)

        npy2_path = path_npy2 + "/" + file_list[i]
        np.save(npy2_path, zoomed_data)
        sys.stdout.write('\r' + f'resize Progress: {i+1}/{max_count}')
    sys.stdout.flush()
    print(" completion of resize task")