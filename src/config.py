import configparser
import os

def get_dataset_path(local_path, server_path):

    if os.path.exists(local_path):
        root_path = local_path
    elif os.path.exists(server_path):
        root_path = server_path
    else:
        raise ValueError('No path is working')

    path_train = os.path.join(root_path, 'MICCAI_BraTS_2019_Data_Training/')
    path_test = os.path.join(root_path, 'MICCAI_BraTS_2019_Data_Validation/')

    return path_train, path_test


def get_configuration(path):

    config = configparser.ConfigParser()

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    config.read(path)

    return config