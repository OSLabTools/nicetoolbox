# Algorithms

- [HrNet & VitPose](#hrnet--vitpose)
- [XGaze_3cams](#xgaze_3cams)

<br>


## HrNet & VitPose

These are two algorithms for human pose estimation. The available implementations are:

`HRNet-w48+DARK <https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#topdown-heatmap-hrnet-dark-on-coco-wholebody>`__

`ViTPose-L <https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#topdown-heatmap-vitpose-on-coco>`__.


The algorithms estimate the position of joints in 2D. These 2D estimates
are further refined during post-processing.

In post-processing, the algorithm’s results are filtered using
Savitzky-Golay filter. Filtering the algorithm’s results helps address
the well-known flickering problem in pose estimation, but this comes at
the cost of potentially smoothing out small, meaningful movement
changes.This filtering option is kept optional and controlled by the
[frameworks.mmpose]\ ``filtered`` parameter in detectors_config.toml.If
``filtered`` set to true, the ``window_length`` and ``polyorder``
parameters can be used to fine-tune it. For more details see
```scipy.signal.savgol_filter`` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html#scipy-signal-savgol-filter>`__.

Joint estimations with a confidence score below 0.60 are marked as
missing because they often indicate an occluded joint or an incorrect
estimate. After replacing these likely incorrect estimations with
missing values, linear interpolation is applied to the data points
between the last two non-missing estimates of the joint. If the interval
is greater than 1/3 of a second (#TODO: apply as 1/FPS currently
hardcoded as 10 frames), the joint positions are left as empty.

With calibrated stereo cameras, the 3D positions of the body joints are
determined using the triangulation method.



## XGaze_3cams

A version of [ETH-XGAZE](https://github.com/xucong-zhang/ETH-XGaze).

The algorithm can track gaze in 3D even with a single camera. When more than one camera is used, the algorithm aggregates the results from each camera that can detect gaze. Gaze direction results of the algorithm are further refined during post-processing using Savitzky-Golay filter (for more details see ```scipy.signal.savgol_filter`` <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html#scipy-signal-savgol-filter>`__)

The algorithm requires the intrinsics parameters of the used cameras. 

