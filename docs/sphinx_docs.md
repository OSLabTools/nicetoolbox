# Using Sphinx for Documentation

Sphinx is a powerful tool for generating documentation from reStructuredText files. In this guide,
we will walk you through the process of using Sphinx to create professional-looking documentation
for your project.

<!-- TOC -->
- [Using Sphinx for Documentation](#using-sphinx-for-documentation)
  - [Installation](#installation)
  - [Creating a New Sphinx Project](#creating-a-new-sphinx-project)
  - [Generating Simple API Documentation](#generating-simple-api-documentation)
  - [Autosummary](#autosummary)
    - [How does it work?](#how-does-it-work)
    - [How to set it up?](#how-to-set-it-up)
  - [Configuring `conf.py`](#configuring-confpy)
    - [(A) Change Project Source](#a-change-project-source)
    - [(B) Project Information](#b-project-information)
    - [(C) General Configuration](#c-general-configuration)
      - [Extensions](#extensions)
      - [Fix for failing imports](#fix-for-failing-imports)
      - [Setting Extension Variables](#setting-extension-variables)
      - [How to exclude patterns from the documentations](#how-to-exclude-patterns-from-the-documentations)
      - [Templates (Necessary for autosummary)](#templates-necessary-for-autosummary)
    - [(D) Options for HTML Output](#d-options-for-html-output)
      - [Apply Read-The-Docs Theme](#apply-read-the-docs-theme)
      - [Override CSS Settings](#override-css-settings)
  - [Markdown Support](#markdown-support)
  - [View your local documentation](#view-your-local-documentation)

## Installation

Install Sphinx using pip inside the project environment:

```bash
pip install sphinx
pip install myst_parser            # (for markdown support)
pip install sphinx_rtd_theme       # (read-the-docs theme)
```

## Creating a New Sphinx Project

First create a new folder for all your documentation files named `./docs` and navigate into it.
To start a new Sphinx project, you can use the `sphinx-quickstart` command.

```bash
sphinx-quickstart
```

During the setup process, you will be asked to provide information such as the project name,
version, author, and other details. Once you have completed the setup, Sphinx will generate
a set of files and directories for your documentation:

After running `sphinx-quickstart`, the following files and directories will be generated
in the `./docs` folder:

- `conf.py`: The main configuration file for your Sphinx project.
- `index.rst`: The main documentation file that serves as the entry point for your documentation.
- `Makefile`: The makefile responsible for building the documentation
- `make.bat`: The equivalent for windows users. TODO: Rewrite such that dirs deleted before
  building to prevent conflicts.
- `_build/`: The directory where the generated documentation will be stored.
- `_static/`: The directory for static files such as CSS and JavaScript.
- `_templates/`: The directory for custom .rst templates.

**Important:** Given the suggested structure of the project and the `./docs` folder, we have to
change the source path from where Sphinx will start building the documentation.
See [Change Project Source](#a-change-project-source) for a quick guide on setting up the path in
`conf.py`.

## Generating Simple API Documentation

If you have existing Python code that you want to document, Sphinx provides a tool called
`sphinx-apidoc` to automatically generate API documentation from your code. In the current
version of Sphinx (v7.4.5) and to my knowledge, it is only possible to generate `.rst` files
automatically of a single flat module or package (non recursive). It is required to enable
the autodoc extension in the `conf.py` file to use `sphinx-apidoc`. See [Extensions](#extensions)
for a quick guide on setting up the autodoc extension.

To use `sphinx-apidoc` run the following command in the `./docs` folder:

```bash
sphinx-apidoc -o <OUTPUT_PATH> <MODULE_PATH> --ext-autodoc
sphinx-apidoc -o . .. --ext-autodoc
```

This command will scan your project directory and generate `.rst` files for each
file. Finally, it creates a file called modules.rst. This file contains a list of all
modules in your project and links to the individual `.rst` files. Simply add the following
line to the `index.rst` file to include the modules:

```rst
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
```

> Note: In the following section, we will explain how to set up the `autosummary` extension to generate
more sophisticated documentation. *Setting up the `toctree` in `index.rst` will change accordingly.*

## Autosummary

Since the `sphinx-apidoc` command only generates `.rst` files for a single module or package,
and is not recursive, we need to use the `autosummary` extension to generate summaries for
all modules, classes, and functions in our project. There is a autosummary tutorial from
[James Leedham on GitHub](https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion)
that explains how to set up the autosummary extension for recursive documentation
([Forum Post](https://stackoverflow.com/questions/2701998/automatically-document-all-modules-recursively-with-sphinx-autodoc)).
NICE Toolbox uses the same concepts.

### How does it work?

The `autosummary` extension generates summary tables for modules, classes, and functions.
The user can set a folder as starting point to then recursively generate summaries for all
modules, classes, and functions in the project. The extension uses the `autosummary_generate`
variable in the `conf.py` file to enable the feature. The `autosummary` extension requires
the `autodoc` extension to be enabled as well.

### How to set it up?

1. A custom template for classes and modules is necessary to display the summaries correctly.
   The templates must be placed in the `./docs/_templates` folder.
   See [Templates](#templates-necessary-for-autosummary).
2. To enable the `autosummary` extension, add the extension to the `conf.py` file.
   See [Extensions](#extensions).
3. Set the `autosummary_generate` variable to `True` in the `conf.py` file.
   See [Setting Extension Variables](#setting-extension-variables).
4. Python packages contain a `__init__.py` file to mark the directory as a package. Thus, the
   `__init__.py` file should be present in all directories that contain modules to be documented
   recursively.
5. Failing imports can lead to missing files in the documentation. To prevent this, you can set
   mock imports for all import statements that lead to errors. This is especially useful when
   the project is split into different python environments. Sphinx operates in a single environment
   and might not be able to run all import statements. See [Fix for failing imports](#fix-for-failing-imports).
6. Create a new directory in the `./docs` folder called `_autosummary` to store the generated
   summary files.

Summaries are .rst files that contain a list of all modules, classes, and functions in the
project for each directory, class and function. All these files are generated automatically
and saved in the `./docs/_autosummary` folder. The summaries are then included in the `index.rst`
file using the `autosummary` directive. For each starting point to recursively create a
API reference, create a new `.rst` file in the `./docs` folder and include the following code:

```rst
.. autosummary::
   :toctree: _autosummary                 - links to the generated summaries in the _autosummary folder
   :template: custom-module-template.rst  - custom template for modules necessary for displaying the docs
   :recursive:                            - recursive flag

   <MODULE>                               - name of the module to be documented (e.g. detectors, visual)
```

In the NICE Toolbox, we created 3 starting points for recursive documentation: API references for
the detectors, evaluation and visual source code directories. Finally, include the new `.rst` files
in the `index.rst` file using the `toctree` directive.

```rst
.. toctree::
   :hidden:

   Home page <self>                          - link to the index.rst file
   Overview of Project <README>              - main documentation files(.md or .rst)
   Installation <installation>               - |
   Getting started <getting_started>         - |
   Tutorial <tutorials>                      - |
   Detectors api <_autosummary/detectors>    - starting points for recursive documentation
   Evaluation api <_autosummary/evaluation>  - |
   Visualization api <_autosummary/visual>   - |
```

The `toctree` directive is used to create a table of contents for the documentation. At the same time
it represents the navigation bar on the left side of the documentation page. The `:hidden:`
flag hides the table of contents on the Home Page. The `self` flag links to the
`index.rst` file. The `README`, `installation`, `getting_started`, and `tutorials` files are markdown
.rst files and them main documentation files for the project. They are inside the `./docs` folder or
have to be selected accordingly. The `detectors.rst`, `evaluation.rst`, and `visual.rst` files are the
starting points for the recursive documentation.

## Configuring `conf.py`

The `conf.py` file is the main configuration file for your Sphinx project. It contains various
settings that control the behavior of Sphinx and the appearance of your documentation. For the
full list of built-in configuration values, see the
[documentation](https://www.sphinx-doc.org/en/master/usage/configuration.html).

### (A) Change Project Source

If extensions (or modules to document with autodoc) are in another directory,
add these directories to sys.path here. If the directory is relative to the
documentation root, use os.path.abspath to make it absolute, like shown here.

```python
import os
import sys

sys.path.insert(0, os.path.abspath('..')) # Parent dir of ./docs folder should be the project top level dir
```

### (B) Project Information

Basic information about the project. This was already set during sphinx-quickstart and can be
changed again. Go to [this link](https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information)
for more information.

```python
project = 'NICE Toolbox'
copyright = ''
author = ''
release = ''
```

### (C) General Configuration

Some of the important settings in `conf.py` include:

#### Extensions

This setting allows you to enable or disable Sphinx extensions. Extensions provide additional
functionality to Sphinx, such as support for different markup languages or themes.

```python
extensions = [
    'sphinx.ext.napoleon',        # Support for NumPy and Google style docstrings
    'sphinx.ext.autodoc',         # Core Sphinx library for auto html doc generation from docstrings
    'sphinx.ext.autosummary',     # Create summary tables for modules/classes/methods recursively
    'sphinx.ext.intersphinx',     # Link to other project's documentation
    'sphinx.ext.viewcode',        # Add a link to the Python source code for classes and functions
    'myst_parser',                # Markdown support
    # 'sphinx.ext.todo',          # Include to dos in the documentation
]
```

#### Fix for failing imports

- Sphinx needs to create .rst files for all modules, classes and functions. All the files
to be included for documentation have to be interpreted and compiles though sphinx at some
point.
- Any failed imports or errors lead to the files not being included in the documentation
when using recursive autosummary combined with autodoc. Especially when working with
different pythons envs that are responsible for different modules / code blocks inside
the project, Sphinx is unable to successfully run all import statements (If the environment
activated does not include all dependencies).

> Solution:
You can set mock imports for all import statements that lead to errors:

```python
autodoc_mock_imports = [
    "mmpose.apis", "mmpose", "data", 
    "torch", "torchvision", "face_alignment"
    ] 
```

#### Setting Extension Variables

```python
# Enable autosummary generation
autosummary_generate = True        

# Add __init__ doc (i.e., params) to class summaries
autoclass_content = "both"         

# Remove 'view source code' from top of page
html_show_sourcelink = False       

# If no docstring, inherit from base class
autodoc_inherit_docstrings = True  

# Enable 'expensive' imports for sphinx_autodoc_typehints
set_type_checking_flag = True      

# Fix for missing cross-references for headings in Markdown files
myst_heading_anchors = 7 
```

#### How to exclude patterns from the documentations

The `exclude_pattern` setting allows you to specify patterns for files or directories that
should be excluded from the documentation:

```python
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
```

#### Templates (Necessary for autosummary)

Based on the autosummary tutorial from
[James Leedham](https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion), we need to add
two custom template .rst files for classes and modules. These files are necessary for the autosummary
files to be displayed correctly. Please copy the files
[custom-class-template.rst](https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion/blob/master/docs/_templates/custom-class-template.rst)
and [custom-module-template.rst](https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion/blob/master/docs/_templates/custom-module-template.rst)
and add them to the `_templates` folder. Finally, add the following line to the `conf.py` file:

```python
templates_path = ['_templates']
```

### (D) Options for HTML Output

Here you can apply themes and override CSS settings. For more details, check the
[documentation](https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output).
The setting `html_theme` determines the theme used for generating HTML documentation.
Sphinx comes with several built-in themes, and you can also install third-party themes.

#### Apply Read-The-Docs Theme

Given that `sphinx_rtd_theme` is installed, the following code snipped overrides the standard
sphinx theme:

```python
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:  # only import and set the theme if we're building docs locally
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
```

#### Override CSS Settings

It is possible to override css settings based on a provided css file. The file has to be inside
the `./docs/_static` folder. We applied custom readthedocs css setting that can be found
[here](https://github.com/JamesALeedham/Sphinx-Autosummary-Recursion/blob/master/docs/_static/readthedocs-custom.css):

```python
html_static_path = ['_static']
html_css_files = ["readthedocs-custom.css"]
```

## Markdown Support

Sphinx does not natively support markdown files. However, the `myst_parser` extension allows you
to write markdown files and include them in your Sphinx project. To enable markdown support, install
the pip package and add the following line to the `conf.py` file:

```python
extensions = [
    ...
    'myst_parser',
    ...
]
```

If problems regarding header references occur, you can set the `myst_heading_anchors` variable in
the `conf.py` file. The `myst_heading_anchors` variable specifies the minimum heading level that
should be included in the table of contents. The default value is 2, but you can set it to 7 to
include all headings:

```python
myst_heading_anchors = 7
```

## View your local documentation

To view your documentation locally, navigate to the `./docs` folder and run the following command:

```bash
make html
```

This command will generate the documentation in the `_build` directory. You can then open the
`_build/html/index.html` file in your browser to view the documentation. If you make changes to
the documentation, you can run `make html` again to update the generated files.

> Note: The Makefile is generated by Sphinx and contains various commands for building the documentation.
> I updated the Makefile to delete the `_build` and `_autosummary` directories before building the
> documentation to prevent conflicts (for example old .rst files that are not part of the code anymore).
