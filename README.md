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

## ToDos 


### Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.tuebingen.mpg.de/cschmitt/template-detector/-/settings/integrations)

### Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

### Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)


### Adding to this README

- **Description**: 
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

- **Badges**:
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

- **Visuals**:
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

- **Usage**:
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

- **Support**:
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

- **Contributing**:
State if you are open to contributions and what your requirements are for accepting them.
For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.
