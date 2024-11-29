# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NICE Toolbox'
copyright = '2024, oslab'
author = 'oslab'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Support for NumPy and Google style docstrings
    'sphinx.ext.napoleon',  
    # Core Sphinx library for auto html doc generation from docstrings
    'sphinx.ext.autodoc',  
    # Create neat summary tables for modules/classes/methods etc
    'sphinx.ext.autosummary',  
    # Link to other project's documentation (see mapping below)
    'sphinx.ext.intersphinx',  
    # Add a link to the Python source code for classes, functions etc.
    'sphinx.ext.viewcode',  
    # Markdown support
    'myst_parser', 
    # to add video directly in the documentation
    "sphinxcontrib.video",
    # # Include todos in the documentation
    # 'sphinx.ext.todo',  
    # # Automatically document param types (less noise in class signature)
    # 'sphinx_autodoc_typehints', 
]

autodoc_mock_imports = [
    "mmpose.apis", "mmpose", "data", 
    "torch", "torchvision", "face_alignment"
    ] # fix for failing imports using sphinx.ext.autodoc

# Turn on sphinx.ext.autosummary
autosummary_generate = True  
# Add __init__ doc (ie. params) to class summaries
autoclass_content = "both"  
# Remove 'view source code' from top of page (for html, not python)
html_show_sourcelink = False  
# If no docstring, inherit from base class
autodoc_inherit_docstrings = True  
# Enable 'expensive' imports for sphinx_autodoc_typehints
set_type_checking_flag = True  
# fix for missing cross-references for headings
myst_heading_anchors = 7 

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Readthedocs theme
# on_rtd is whether on readthedocs.org, this line of code grabbed from 
# docs.readthedocs.org...
# on_rtd = os.environ.get("READTHEDOCS", None) == "True"
# if not on_rtd:  # only import and set the theme if we're building docs locally
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    
html_theme_options = {
    'collapse_navigation': False,  # Set to False to prevent collapsing of submenus
    'titles_only': False,          # If set to True, only titles are shown
}

html_static_path = ['_static']
html_css_files = ["readthedocs-custom.css"] # Override some CSS settings


