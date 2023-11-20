"""Configuration file for the Sphinx documentation builder."""

# -- Project information

project = "fl-project-template"
copyright = """2023,
Flower Authors <hello@flower.dev>,
Alexandru-Andrei Iacob <aai30@cam.ac.uk>,
Lorenzo Sani <ls985@cam.ac.uk>"""
author = """Flower Authors <hello@flower.dev>, Alexandru-Andrei Iacob <aai30@cam.ac.uk>,
Lorenzo Sani <ls985@cam.ac.uk>"""

release = "1.0"
version = "1.0.0"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
