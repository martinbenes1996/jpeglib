
# remove previous releases
rm -rf build/ dist/ src/jpeglib.egg-info/ __pycache__/
rm -rf src/jpeglib/cjpeglib/*.so

# compile
python setup.py bdist --verbose
retVal=$?
if [ $retVal -ne 0 ]; then
    exit $retVal
fi

# get dynamic libs
cp $(find build/lib* -maxdepth 0)/jpeglib/cjpeglib/*.so src/jpeglib/cjpeglib/

python run.py
