# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'jpeglib'
copyright = '2021, Martin Benes'
author = 'Martin Benes'

release = '0.14'
version = '0.14.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosectionlabel'
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output
html_theme = 'sphinx_rtd_theme'  # 'alabaster'

# -- Options for EPUB output
epub_show_urls = 'footnote'
