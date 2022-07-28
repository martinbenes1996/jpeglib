
#if ! ls jpeglib/cjpeglib/*.so 1> /dev/null 2>&1 ; then
    
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

#fi

# run
python tests/test.py
retVal=$?
if [ $retVal -ne 0 ]; then
    exit $retVal
fi