
# remove previous releases
rm -rf build/ dist/ stegojpeg.egg-info/ __pycache__/
rm -rf jpeglib/cjpeglib/*.so

# compile
python setup.py bdist --verbose
retVal=$?
if [ $retVal -ne 0 ]; then
    exit $retVal
fi

# get dynamic libs
cp $(find build/lib* -maxdepth 0)/jpeglib/cjpeglib/*.so jpeglib/cjpeglib/

python run.py
#python -c 'import jpeglib; im = jpeglib.read_jpeg_dct("examples/IMG_0311.jpeg"); print(im.qt)'