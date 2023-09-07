"""

"""

import os
from oslab_utils.logging import log_configs
from oslab_utils.config import config_fill_auto, config_fill_placeholders, load_config


class Configuration:
    def __init__(self, config_abstract_file, machine_specifics_file):
        # load experiment config dicts - these might contain placeholders
        self.config_abstract = load_config(config_abstract_file)
        self.machine_specifics = load_config(machine_specifics_file)
        self.machine_specifics.update(dict(pwd=os.getcwd()))

        # fill placeholders
        self.config = config_fill_auto(self.config_abstract)
        self.localized_config = config_fill_placeholders(
                self.config, self.machine_specifics)
        self.localized_config = config_fill_placeholders(
                self.localized_config, self.localized_config['io'])

    def get_localized_config(self):
        return self.localized_config

    def save_experiment_config(self, output_folder):
        # save all experiment configurations
        log_configs(dict(config=self.config,
                         machine_specifics=self.machine_specifics),
                    output_folder)
