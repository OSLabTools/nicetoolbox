# Understanding the Calibration files


WORK IN PROGRESS...


This is an extension of the dataset directory. Calibration parameters for all cameras are provided as a nested dictionary, its file path is saved in `./detectors/configs/dataset_properties.toml` under `path_to_calibrations` as `.json` or `.npz`.

**Example:**

```json
{
  "camera_1": {
    "camera_name": "camera_1",
    "image_size": [1000, 700],
    "intrinsic_matrix": [
      [1193.6641930950077, 0.0, 503.77365693839107],
      [0.0, 1193.410339778219, 352.12891433016244],
      [0.0, 0.0, 1.0]
    ],
    "distortions": [-0.1412521384983322, 0.14702510007618264, 0.00010429739735286396, -0.0004644593818576435],
    "rvec": [0.0, 0.0, 0.0],
    "rotation_matrix": [
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0]
    ],
    "translation": [[0.0], [0.0], [0.0]],
    "extrinsics_matrix": [
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0, 0.0]
    ],
    "projection_matrix": [
      [1193.6641930950077, 0.0, 503.77365693839107, 0.0],
      [0.0, 1193.410339778219, 352.12891433016244, 0.0],
      [0.0, 0.0, 1.0, 0.0]
    ]
  },
  "camera_2": {
    // camera parameters...
  }
}
```

- `camera_name` (str): the name of the camera, same as dict key.

- `image_size` (list): the width and the height of the image that the camera captures. The list format is [image_width, image_height]

- `intrinsics_matrix` (list-of-lists): This matrix, also known as the camera matrix, contains intrinsic parameters of the camera.
The list format is: `[[fx,s,cx],[0,fy,cy],[0,0,1]]`. It is structured as follows:
  - `fx/fy`: focal length (if fx/fy is not specified, f=fx, fy=fx*ar)
  - `s`: skewness, mostly 0
  - `cx/cy`: principle points

- `distortions` (list): The distortion coefficients which correct for lens distortion in the captured images. These coefficients follow the OpenCV model and usually include [k1, k2, p1, p2, k3]
  - `k1, k2, k3` : Radial distortion coefficients.
  - `p1, p2`: Tangential distortion coefficients.

- `rvec` (list): rotation vector - `cv2.rodrigues(rotation_matrix)` - a 3-element vector, a compact representation of rotation matrix.
Its direction represents the axis of rotation and whose magnitude represents the angle of rotation in radians. also known as axis-angle representation.

- `rotation_matrix` (list-of-lists): 3x3 rotation matrix `R` - `cv2.rodrigues(rvec)`

- `translation` (list-of-lists): the translation `t` of the camera from the origin of the coordinate system.

- `extrinsics_matrix` (list-of-lists): This is a 3x4 matrix that combines the rotation matrix and translation vector to describe the camera's position and orientation in space. It is optained by stacking rotation matrix with tranlsation vector: `[R|t]`.

- `projection_matrix` (list-of-lists): 4x4 matrix that projects 3D points in the camera's coordinate system into 2D points in the image coordinate system.
It is obtained by multiplying the intrinsic matrix by the extrinsic matrix: `np.matmul(intrinsic_matrix, extrinsic_matrix)`.

The extrinsic parameters represent a rigid transformation from 3-D world coordinate system to the 3-D camera’s coordinate system.
The intrinsic parameters represent a projective transformation from the 3-D camera’s coordinates into the 2-D image coordinates.
For more information, see a great introduction from [MathWorks](https://de.mathworks.com/help/vision/ug/camera-calibration.html).




## Creating the calibration file


If loading an existing calibration file, the calibration dictionary may look like this in the case of using OpenCV's camera matrix representation:
```toml
[<session_ID>__<sequence_ID>.<camera_name>]
camera_name = "<camera_name>"
image_size = [ <width>, <height>]
mtx = [ [<f_x>, 0.0, <c_x>], [0.0, <f_y>, <c_y>], [ 0.0, 0.0, 1.0]]
dist = [ 0.0, 0.0, 0.0, 0.0, 0.0]
rvec = [ 0.0, 0.0, 0.0]
tvec = [ [ 0.0], [ 0.0], [ 0.0]]
```



Try our calibration converter GUI!

On linux:

```bash
cd isa-tool/
source envs/isa2/bin/activate
python utils/calibration_gui/calibration_converter.py
```

The calibration converter offers multiple options to create, load, or change a calibration file for the NICE toolbox. It outputs the calibration in two files: `calibrations.npz` which is required to run the NICE toolbox and `calibrations.json` which displays the same calibration data in a human-readable (and changeable) file.  
##TODO! Change json to toml


**For a single camera**
In case your dataset has a single camera only (no multi-view setup), feel free to leave the rotation matrix (usually `R` or `rvec`) and the translation matrix (commonly denoted with `t` or `tcev`) to the defaults of identity:

```
"rotation_matrix" or "R":       [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
"vector" or "rvec":             [[0.0], [0.0], [0.0]]
"translation", "t", or "tvec":  [0.0, 0.0, 0.0]
```

Similarly, if you do not know the distortion coefficients, set them to `0.0`:
```
"distortions" or "d":           [0.0, 0.0, 0.0, 0.0, 0.0]
```
