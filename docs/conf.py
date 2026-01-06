# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

# -- Project information -----------------------------------------------------

project = "PynPoint"
copyright = "2026, Tomas Stolker & Markus Bonse"
author = "Tomas Stolker & Markus Bonse"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_automodapi.automodapi",
    "nbsphinx"
]

numpydoc_show_class_members = False

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.

exclude_patterns = ["_build",
                    "Thumbs.db",
                    ".DS_Store",
                    "tutorials/.ipynb_checkpoints/*"]

# Disable notebook timeout
nbsphinx_timeout = -1

# Allow errors from notebooks
nbsphinx_allow_errors = True

autoclass_content = "both"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_book_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#

html_theme_options = {
    "path_to_docs": "docs",
    "repository_url": "https://github.com/PynPoint/PynPoint",
    "repository_branch": "main",
    "launch_buttons": {
        "notebook_interface": "jupyterlab",
    },
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}

html_context = {
    "github_user": "PynPoint",
    "github_repo": "PynPoint",
    "github_version": "main",
    "doc_path": "docs",
}

html_static_path = ["_static"]

html_logo = "_static/logo.png"
html_favicon = "_static/favicon.png"
html_search_language = "en"
