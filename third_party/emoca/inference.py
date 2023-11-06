import os
import sys
import third_party.file_handling as fh
import logging
import numpy as np

sys.path.append("./third_party/emoca/emoca")
sys.path.append("./third_party")
from third_party.emoca.emoca.gdl_apps.EmotionRecognition.utils.io import load_model
from third_party.emoca.emoca.gdl.datasets.ImageTestDataset import TestData
from pathlib import Path
from tqdm import auto
from third_party.emoca.emoca.gdl.utils.other import get_path_to_assets
from third_party.emoca.emoca.gdl_apps.EmotionRecognition.demos.test_emotion_recognition_on_images import save_images
from third_party.emoca.emoca.gdl.datasets.AffectNetDataModule import AffectNetExpressions


def main(config):
    path_to_models = get_path_to_assets() / "EmotionRecognition" / "face_reconstruction_based"
    model_name = 'EMOCA-emorec'

    # 1) Load the model
    model = load_model(Path(path_to_models) / model_name)
    model.cuda()
    model.eval()

    expressions = {name: [] for name in config["camera_names"]}
    for camera_name in config["camera_names"]:
        logging.info(f'Camera - {camera_name}')
        input_folder = os.path.join(config["input_data_folder"], camera_name)
        fh.assert_and_log(
                os.path.exists(input_folder),
                f"Could not find data folder for camera: '{camera_name}'")

        out_folder = os.path.join(config['out_folder'], camera_name)
        os.makedirs(out_folder, exist_ok=True)

        # 2) Create a dataset
        dataset = TestData(input_folder, face_detector="fan", max_detection=20)

        ## 3) Run the model on the data
        for i in auto.tqdm(range(len(dataset))):
            batch = dataset[i]
            batch["image"] = batch["image"].cuda()
            output = model(batch)

            expression = np.concatenate(
                    [output[name].cpu().detach().numpy().flatten() for name
                     in ['valence', 'arousal', 'expr_classification']], axis=0)
            expressions[camera_name].append(expression)
            save_images(batch, output, out_folder)

    #  save as hdf5 file
    fh.save_to_hdf5(
            [np.array(expressions[cam]) for cam in config["camera_names"]],
            config["camera_descr"],
            os.path.join(config["result_folder"], f"expressions.hdf5"),
            index=os.listdir(input_folder)
    )
    fh.save_toml(
            {'expression_names':
                 ['valence', 'arousal'] +
                 [expr.name.lower() for expr in AffectNetExpressions]
             },
            os.path.join(config["result_folder"], f"expressions.toml"))


if __name__ == '__main__':
    config_path = sys.argv[1]
    config = fh.load_config(config_path)
    main(config)
