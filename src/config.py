import configparser
import os
import time
from src.logging_conf import logger


def create_directory(dir):
    logger.debug(f"Create directory: {dir}")
    if not os.path.exists(dir):
        os.makedirs(dir)

def check_path_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path

def get_dataset_path(local_path, server_path,):

    if os.path.exists(local_path):
        root_path = local_path
    elif os.path.exists(server_path):
        root_path = server_path
    else:
        raise ValueError('No path is working')
    return root_path


class BratsConfiguration:

    def __init__(self, path):

        self.config = configparser.ConfigParser()
        self.path = check_path_exists(path)
        self.config.read(self.path)
        self.prepare_parameters()

    def prepare_parameters(self):
        create_directory(f'{self.config.get("basics", "tensorboard_logs")}_{round(time.time())}')
        create_directory(self.config.get("model", "model_path"))


        self.config["dataset"]["root_path"] = get_dataset_path(self.config.get("dataset", "dataset_root_path_local"),
                                                               self.config.get("dataset", "dataset_root_path_server"))

        self.config["dataset"]["path_train"] = os.path.join(self.config["dataset"]["root_path"], "train")
        self.config["dataset"]["path_val"] = os.path.join(self.config["dataset"]["root_path"], "validation")
        self.config["dataset"]["train_csv"] = os.path.join(self.config["dataset"]["path_train"], self.config.get("dataset", "train_csv"))
        self.config["dataset"]["val_csv"] = os.path.join(self.config["dataset"]["path_val"], self.config.get("dataset", "val_csv"))

        self.config["dataset"]["batch_size"] = str(self.config.getint("dataset", "n_patients_per_batch") * self.config.getint("dataset", "n_patches"))
        self.patch_size = tuple([int(item) for item in self.config.get("dataset", "patch_size").split("\n")])


    def get_model_config(self):
        return self.config["model"]

    def get_dataset_config(self):
        return self.config["dataset"]

    def get_basic_config(self):
        return self.config["basics"]

