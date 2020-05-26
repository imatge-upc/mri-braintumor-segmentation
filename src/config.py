import configparser
import os
import time
from src.logging_conf import logger
from src.dataset import io_utils


def create_directory(dir):
    logger.debug(f"Create directory: {dir}")
    if not os.path.exists(dir):
        os.makedirs(dir)

def check_path_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


class BratsConfiguration:

    def __init__(self, path):

        self.config = configparser.ConfigParser()
        self.path = check_path_exists(path)
        self.config.read(self.path)
        self.prepare_parameters()

    def prepare_parameters(self):
        create_directory(f'{self.config.get("basics", "tensorboard_logs")}_{round(time.time())}')
        create_directory(self.config.get("model", "model_path"))

        self.config["dataset"]["path_train"], self.config["dataset"]["path_test"] = \
            io_utils.get_dataset_path(self.config.get("dataset", "dataset_root_path_local"),
                                      self.config.get("dataset", "dataset_root_path_server"))

        self.config["dataset"]["batch_size"] = str(self.config.getint("dataset", "n_patients_per_batch") * self.config.getint("dataset", "n_patches"))
        self.patch_size = tuple([int(item) for item in self.config.get("dataset", "patch_size").split("\n")])


    def get_model_config(self):
        return self.config["model"]

    def get_dataset_config(self):
        return self.config["dataset"]

    def get_basic_config(self):
        return self.config["basics"]

