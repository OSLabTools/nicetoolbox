import os
import sys
import logging

sys.path.append("./third_party/active_speaker")
sys.path.append("./third_party")
from ASC_data_preparation import data_preparation
from ASC_audiovisual_feature_extraction import feature_extraction
from GraViT_generate_graph import generate_graphs_in_parallel
from GraViT_evaluation import evaluate
import third_party.file_handling as fh


def main(config):
    """ Run inference of the method on the pre-loaded image
    """
    # initialize logger
    logging.basicConfig(filename=config['log_file'], level=config['log_level'],
                        format='%(asctime)s [%(levelname)s] %(module)s.%(funcName)s: %(message)s')
    logging.info('\n\nRUNNING active speaker detection!')

    # check that the data was created
    fh.assert_and_log(config['snippets_list'] is not None,
        f"Active speaker detection: Please specify 'input_data_format: 'snipp"
        f"ets' in config. Currently it is '{config['input_data_format']}'.")

    # pre-process data
    data_preparation(config)

    # extract audio-visual features
    feature_extraction(config)

    # build graph
    generate_graphs_in_parallel(config)

    # detect active speaker
    evaluate(config)

    logging.info('\nActive speaker detection COMPLETED!')


if __name__ == '__main__':
    config_path = sys.argv[1]
    config = fh.load_config(config_path)
    main(config)
