"""

"""

import os
import glob
import logging
from oslab_utils.logging_utils import log_configs
from oslab_utils.config import config_fill_auto, config_fill_placeholders, load_config
import oslab_utils.check_and_exception as exc


def get_top_level_dict(dictionary):
    return {k: v for k, v in dictionary.items() if not isinstance(v, dict)}


def compare_configs(config1, config2, log_fct=print, config_names=''):
    shared_keys = set(config1.keys()).intersection(set(config2.keys()))

    # do the configs contain different sets of keys?
    all_keys = set(config1.keys()).union(set(config2.keys()))
    non_shared_keys = all_keys.difference(shared_keys)
    if len(non_shared_keys) != 0:
        log_fct(f"Configs {config_names} differ in their keys! "
                f"Key(s) '{non_shared_keys}' only exist(s) for one of them.")

    # do the values match?
    keys_different_values = [k for k in shared_keys if config1[k] != config2[k]]
    if len(keys_different_values) != 0:
        log_fct(f"Configs {config_names} contain different values for keys "
                f"'{keys_different_values}!")


class Configuration:
    def __init__(self, config_abstract_file, machine_specifics_file):
        # load experiment config dicts - these might contain placeholders
        self.config_abstract = load_config(config_abstract_file)
        self.machine_specifics = load_config(machine_specifics_file)
        self.machine_specifics.update(dict(pwd=os.getcwd()))

        #
        self.check_config()

        # add dataset settings
        dataset_name = self.config_abstract['io']['dataset_name']
        self.config_abstract['io'].update(self.config_abstract['datasets'][dataset_name])
        del self.config_abstract['datasets']

        # fill placeholders
        self.config = config_fill_auto(self.config_abstract)
        self.localized_config = config_fill_placeholders(
                self.config, self.machine_specifics)
        self.localized_config = config_fill_placeholders(
                self.localized_config, self.localized_config['io'])
        config_level_one = get_top_level_dict(self.localized_config)
        self.localized_config = config_fill_placeholders(
                self.localized_config, config_level_one)

    def check_config(self):
        try:
            exc.check_options(
                self.config_abstract['run_mode'], str,
                ["experiment", "development", "production"])
        except (TypeError, ValueError):
            logging.exception(
                f"Unsupported input for '{self.config_abstract['run_mode']}'. "
                f"Options are 'experiment', 'development', 'production'.")
            raise

        if self.config_abstract['run_mode'] == 'production' and __debug__:
            logging.error(
                "In 'production' mode, this script is meant to be run "
                "with environment variable 'PYTHONOPTIMIZE=1' or, "
                "equivalently, 'python -0' option for optimized performance.")



    def get_localized_config(self):
        return self.localized_config

    def save_experiment_config(self, output_folder):
        # save all experiment configurations
        log_configs(dict(config=self.config,
                         machine_specifics=self.machine_specifics),
                    output_folder,
                    file_name=f"config_<time>")

    def check_config_consistency(self, folder):
        match self.localized_config['run_mode']:
            case "experiment":
                log_fct = logging.error
            case _:
                log_fct = logging.info

        saved_config_files = glob.glob(os.path.join(folder, 'config_*.toml'))
        if len(saved_config_files) != 0:
            for saved_config_file in saved_config_files:
                config_names = f'current and loaded ' \
                               f'({os.path.basename(saved_config_file)})'
                saved_config = load_config(saved_config_file)

                # compare top level dicts
                compare_configs(get_top_level_dict(saved_config['config']),
                                get_top_level_dict(self.localized_config),
                                logging.error, config_names)

                # compare machine specifics
                compare_configs(saved_config['machine_specifics'],
                                self.machine_specifics, logging.info,
                                config_names)

                # compare io
                compare_configs(saved_config['config']['io'],
                                self.config['io'], log_fct, config_names)


