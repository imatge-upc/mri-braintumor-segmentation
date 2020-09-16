import configparser
import os
import time
from src.logging_conf import logger


def create_directory(directory):
    logger.debug(f"Create directory: {directory}")
    if not os.path.exists(directory):
        os.makedirs(directory)


def check_path_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def get_correct_path(local_path, server_path):
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

        self.config["model"]["model_path"] = get_correct_path(self.config.get("model", "model_path_local"),
                                                              self.config.get("model", "model_path_server"))

        train = self.config.getboolean("basics", "train_flag")
        resume = self.config.getboolean("basics", "resume")

        if train:
            if resume:
                model_name, _ = os.path.split(self.config.get("model", "checkpoint"))
            else:
                model_name = f"model_{round(time.time())}"

            logger.info("Create model directory and save configuration")
            self.config["basics"]["tensorboard_logs"] = os.path.join(self.config.get("basics", "tensorboard_logs"),
                                                                     model_name)
            self.config["model"]["checkpoint"] = os.path.join(self.config.get("model", "model_path"),
                                                              self.config.get("model", "checkpoint"))

            create_directory(self.config.get("basics", "tensorboard_logs"))
            self.config["model"]["model_path"] = os.path.join(self.config.get("model", "model_path"), model_name)

            create_directory(self.config["model"]["model_path"])
            # save current configuration there
            with open(os.path.join(self.config["model"]["model_path"], "config.ini"), 'w') as configfile:
                self.config.write(configfile)

        if train:
            sampling_method = self.config["dataset"]["source_sampling"].split(".")[-1]
        else:
            sampling_method = self.config["dataset"]["sampling_method"].split(".")[-1]

        # SETTING PATH!
        self.config["dataset"]["root_path"] = get_correct_path(self.config.get("dataset", "dataset_root_path_local"),
                                                               self.config.get("dataset", "dataset_root_path_server"))

        self.config["dataset"]["path_train"] = os.path.join(self.config["dataset"]["root_path"],
                                                            self.config["dataset"]["dataset_train_folder"],
                                                            sampling_method)
        self.config["dataset"]["path_val"] = os.path.join(self.config["dataset"]["root_path"],
                                                          self.config["dataset"]["dataset_val_folder"],
                                                          sampling_method)

        self.config["dataset"]["path_test"] = os.path.join(self.config["dataset"]["root_path"],
                                                           self.config["dataset"]["dataset_test_folder"],
                                                           sampling_method)

        self.config["dataset"]["train_csv"] = os.path.join(self.config["dataset"]["path_train"],
                                                           self.config.get("dataset", "train_csv"))
        self.config["dataset"]["val_csv"] = os.path.join(self.config["dataset"]["path_val"],
                                                         self.config.get("dataset", "val_csv"))
        self.config["dataset"]["test_csv"] = os.path.join(self.config["dataset"]["path_test"],
                                                          self.config.get("dataset", "test_csv"))

        if "batch_size" not in self.config["dataset"]:
            self.config["dataset"]["batch_size"] = str(
                self.config.getint("dataset", "n_patients_per_batch") * self.config.getint("dataset", "n_patches"))

        self.patch_size = tuple([int(item) for item in self.config.get("dataset", "patch_size").split("\n")])

    def get_model_config(self):
        return self.config["model"]

    def get_uncertainty_config(self):
        return self.config["uncertainty"]

    def get_dataset_config(self):
        return self.config["dataset"]

    def get_basic_config(self):
        return self.config["basics"]
