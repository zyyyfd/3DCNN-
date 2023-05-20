


import argparse
import os

def parse_args(parser):
    
    #文件路径
    #xsd存放原始的xsd数据集。cif存放xsd转换好的cif文件。
    #npy1存放网格化之后的cif文件。npy2存放尺寸缩放好的npy1文件，npy2文件及是训练用文件。
    #labels是吸附能数据。label是转换好的吸附能数据，用于训练及验证
    parser.add_argument('--path_xsd', type=str, default='./data/original xsd')
    parser.add_argument('--path_cif', type=str, default='./data/original cif')
    parser.add_argument('--path_npy1', type=str, default='./data/original npy1')
    parser.add_argument('--path_npy2', type=str, default='./data/original npy2')
    parser.add_argument('--path_label', type=str, default='./data/original label')
    parser.add_argument('--path_labels', type=str, default='./data/labels')
    parser.add_argument('--model_save_path', type=str, default='./model')
    parser.add_argument('--model_path', type=str, default='./model')
    
    #网格化密度，越高采样越多，精度越好
    parser.add_argument('--grid_density', type=int, default=1)
    
    #npy2的shape，目前我训练用的是32*32*32,同样的越大越精细
    parser.add_argument('--new_shape', type=list, default=[32,32,32])
    
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--lengths', type=float, default=0.9)
    
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=0)
    parser.add_argument('--patience', type=int, default=10)
    
    parser.add_argument('--color_channel', type=int, default=7)
    
    parser.add_argument('--crystal_visualization_path', type=str, default='./data/original npy2/1.npy')
    
    return parser

