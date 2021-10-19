

# remove previous releases
rm -rf build/ dist/ stegojpeg.egg-info/ __pycache__/
# compile
python setup.py sdist --verbose
# publish
python -m twine upload dist/* --verbose