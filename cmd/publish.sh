
# remove previous releases
rm -rf build/ dist/ src/jpeglib.egg-info/ __pycache__/
# compile
python setup.py sdist
# publish
python -m twine upload dist/*