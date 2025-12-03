import json
from collections import OrderedDict, Counter
import random
import argparse
import os
import numpy as np
import glob
import tqdm
import tarfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='demo microgesture')
    parser.add_argument('--data_version', '-v', default="0724",
                        help='data version')
    parser.add_argument('--data_path', '-d', default="/data/dataset/all-data-121922/",
                        help='data path')

    args = parser.parse_args()

    folder_list = os.listdir(args.data_path)

    print(folder_list)
    for folder_name in tqdm.tqdm(folder_list):
        assert folder_name[0] in ["z", "R", "Z", "r"]
        print(folder_name)
        
        my_tar = tarfile.open(os.path.join(args.data_path,folder_name,"Export_py","AhatDepth_synced.tar"))
        my_tar.extractall(os.path.join(args.data_path,folder_name,"Export_py","AhatDepth")) # specify which folder to extract to
        my_tar.close()

        source_folder = os.path.join(args.data_path,folder_name,"Export_py","AhatDepth/mnt/hl2data-westus2/all-data-121922/{}/Export_py/AhatDepth".format(folder_name))
        target_folder = os.path.join(args.data_path,folder_name,"Export_py","AhatDepth")
        cmd = 'mv {}/* {}'.format(source_folder, target_folder)
        #cmd = 'rm -rf {}/mnt'.format(target_folder)
        os.system(cmd)