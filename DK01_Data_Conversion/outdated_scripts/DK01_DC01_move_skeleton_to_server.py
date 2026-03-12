""" Copy labeling/solving skeleton from Remote Desktop and save it on the server. """

import os
import shutil

# ROOT_DIR = 'D:/Shogun/SonE_bvh_conversion/Capture day 1/'
ROOT_DIR = 'D:/Shogun/SonE_bvh_conversion/Elderly/'
SERVER_DIR = '//hest.nas.ethz.ch/green_groups_lmb_public/Projects/NCM/NCM_SonEMS/project_only/04_Students/01_Finished/07_Ahaeflig/Training Data/'
DEST_DIR = 'D:/ZM_Data/'

def move_skeleton_to_server():
    VSS = 'Actor_1.vss'
    VSK = 'Actor_1.vsk'

    for f in os.listdir(ROOT_DIR):
        file_name, file_ext = os.path.splitext(f)
        if file_ext == '':
            # src_path = f'D:/Shogun/SonE_bvh_conversion/Capture day 1/{file_name}/'
            src_path = f'D:/Shogun/SonE_bvh_conversion/Elderly/{file_name}/'
            dst_path = f'//hest.nas.ethz.ch/green_groups_lmb_public/Projects/NCM/NCM_SonEMS/project_only/09_Projects_Alex/Training Data/{file_name}/Skeletons/'
            try:
                os.mkdir(dst_path)
            except OSError as error:
                print(error)

            for file in [VSS, VSK]:
                _src_path = src_path + file
                _dst_path = dst_path + file
                shutil.copy2(_src_path,_dst_path)



def move_white2_to_server():
    FILE = 'White2.bvh'

    for f in os.listdir(ROOT_DIR):
        file_name, file_ext = os.path.splitext(f)
        if file_ext == '':
            try:
                _src_path = ROOT_DIR + f'{file_name}/' + FILE
                _dst_path_server = SERVER_DIR + f'{file_name}/bvh/' + FILE
                _dst_path = DEST_DIR + f'{file_name}/bvh/' + FILE

                shutil.copy2(_src_path,_dst_path)
                shutil.copy2(_src_path,_dst_path_server)
            except OSError as error:
                print(error)


def move_fk_folder_to_server():

    root_dir = 'D:/ZM_Data/'
    
    for f in os.listdir(root_dir):
        file_name, file_ext = os.path.splitext(f)
        
        if file_ext == '':
            src_path = root_dir + f'/{file_name}/fk/'
            dst_path = SERVER_DIR + f'/{file_name}/fk/'

            try:
                shutil.copytree(src_path, dst_path)
            except OSError as error:
                print(error)



move_fk_folder_to_server()