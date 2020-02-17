import json
from bunch import Bunch
import os
from src.utils.dirs import create_dirs


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.summary_dir = os.path.join("./experiments", config.exp_name, "summaries/")
    config.checkpoint_dir = os.path.join("./experiments", config.exp_name, "checkpoints/")
    config.saved_model_dir = os.path.join("./experiments", config.exp_name, "saved_models/")
#     config.images_dir = os.path.join("./experiments", config.exp_name, "images/")
    create_dirs([config.summary_dir, config.checkpoint_dir,config.saved_model_dir])
    return config