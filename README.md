# Template Detector


## Installation

Pip editable installation:
```
pip3 install -e .
```

These are the special files to make the code a package that is pip installable:
- ```__init__.py```
- ```pyproject.toml```


## Documentation

Using 
- docstrings (""" """) for functions, methods and modules
- comments (#) for lines or paragraphs
Stick to PEP8 and NumPy style docstrings.

Automate documentation using [Sphinx](https://www.sphinx-doc.org/en/master/). Setup:
```
sudo apt-get install python3-sphinx
cd template-detector/doc/
# sphinx-quickstart  # if building a documentation from scratch
```
To build the documentation:
```
sphinx-build -b html . ./doc  # or
make html                     # if the 'Makefile' is in the same folder
```
and open `./doc/_build/html/index.html` in your browser.


Create a pdf instead of a html: 
```
make latex
cd _build/latex/
pdflatex detector.tex
```


## Testing

Following examples from [ASPP 2019](https://github.com/cscmt/testing_debugging_profiling/tree/master).
Install py.test on Ubuntu:

```
pip install -U pytest
```
From within the code folder, run
```
pytest -v
```
All test functions that start with `test_` are automatically detected by pytest. We collect them in the `./tests/` folder.
Example tests are given in `./tests/test_main.py`.


## Gitlab CI

Added a CI/CD pipeline on gitlab. Use the template `.gitlab-ci.yml` file with some modifications. 

### Runners

From the official docs: *Runners are processes that pick up and execute CI/CD jobs for GitLab. Register as many runners as you want. You can register runners as separate users, on separate servers, and on your local machine.*

[Install Gitlab runner](https://docs.gitlab.com/runner/install/) on Ubuntu:
```
curl -L "https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.deb.sh" | sudo bash
sudo apt-get install gitlab-runner
```
And [register a runner](https://docs.gitlab.com/runner/register/index.html): 
```
sudo gitlab-runner register
```
- GitLab instance URL: `https://gitlab.tuebingen.mpg.de/`
- Registration token: `GR13489411eCZ5zNzjBb5QsxSiVzD`
- Description for the runner: `Test runner on Caro's machine`
- Tags for the runner (*When you register a runner, its default behavior is to only pick tagged jobs.* ): `pytests`
- optiional maintenace note: empty
- Runner executor (*GitLab Runner implements a number of executors that can be used to run your builds in different environments.*): `shell`

In the CI/CD settings, find your runner and set `Can run untagged jobs` to `Yes`.

Find your local runners' configuration file in `/etc/gitlab-runner/config.toml` (needs sudo rights to see)

### TODO venv vs. conda vs pip install

I did not get it working easily with conda. Instead, I use a python virtual machine on my runner and istall the `requirements.txt`. For that, I installed 
```
sudo apt install python3.8-venv
```
Now these `requirements.txt` double the requirements specified in `pyproject.toml`. This is bad. TO FIX!

## Collaboration

### Naming conventions
- Group names: 
personL, personR, dyad

### Shared data formats
1. keypoints: np.arrays saved as hdf5\  
shape: [num_frames, num_keypoints, 2d/3dcoords] in 2d the third score is confidence level\

2. gaze: np.arrays - saved as hdf5\
shape: [num_frames, 6d] 6d: 3d position, 3d coords\

3. nodding/smiling/etc  -might be\
shape: [num_frames, label]\
shape: [num_frames, num_emotions/num_categories, confidence scores]\
shape: [num_annotation/num_segments, [starttime, endtime, label]]\

### Calibration Mapping
```
{
    'PIS_ID_000':'2020-08-11',
    'PIS_ID_01': '2020-08-11',
    'PIS_ID_02': '2020-08-11',
    'PIS_ID_03': '2020-08-11',
    'PIS_ID_04': '2020-08-12',
    'PIS_ID_05': '2020-08-12',
    'PIS_ID_06': '2020-08-12',
    'PIS_ID_07': '2020-08-12',
    'PIS_ID_08': '2020-08-12',
    'PIS_ID_09': '2020-08-13',
    'PIS_ID_666': '2020-08-13'
}
```
### Used ChAruCo Calibration Board
BOARD_SIZE = (12, 9)
SQUARE_SIZE = 60 mm
DICT_5X5

