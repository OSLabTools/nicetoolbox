# Include a dataset with multiple camera views

This tutorial builds up on the tutorial [Include a dataset with a single camera view](tutorial1_dataset_single_view.md) which explains how to run the NICE Toolbox on datasets that contain videos of a single camera, without multi-view captures. This follow-up tutorial now adds instructions specific for multiple calibrated cameras.

WORK IN PROGRESS...

The main difficulty here is to add the calibration information of all cameras.

<br>

**Contents**

1. [Follow steps xxx to xxx](#1-follow-steps-xxx-to-xxx)
2. [Update the dataset properties](#2-update-the-dataset-properties)
3. [Run the toolbox](#3-run-the-toolbox)

<br>


## 1. Follow steps xxx to xxx

Much of the setup for a multi-view dataset equals the setup for the single-view case. Therefore, please follow the instructions for 
- xzx
- dgsdg




## 2. Update the dataset properties

We assume that the cameras are time-synchronized and calibrated intrinsically and extrinsically. First, add the calibration file as described in tutorial on [calibration conversion](tutorial3_calibration_conversion.md).
To run on multi-view camera input, we now need to review the dataset config `./detectors/configs/dataset_properties.toml`. Make sure to update the following keys for your dataset:

```toml
cam_top = ''              # folder name of a frontal camera view from top (str, optional)
cam_face1 = ''            # folder name of a camera view of one subject's face (str, optional)
cam_face2 = ''            # folder name of a camera view of the second subject's face (str, optional)
subjects_descr = []       # define an identifier for the subjects in each video or frame (list of str)
cam_sees_subjects = {}    # define which camera view records which subject (dict: (cam_name, list of int))
path_to_calibrations = "" # file path with placeholders for the calibration files (str, optional)
```

## 3. Run the toolbox

Please follow the instructions in [5. Run the toolbox](tutorial1_dataset_single_view.md#5-run-the-toolbox) for running the experiment. It will now use all cameras provided and specified in the [dataset_properties](#2-update-the-dataset-properties). Check the log-file to see it working.
