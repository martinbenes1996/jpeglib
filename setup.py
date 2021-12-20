

# versions
import os
__version__ = os.environ.get('VERSION_NEW', '0.6.13')
libjpeg_versions = {
  '6b': (None,60),
  '8d': (None,80),
  '9d': (None,90),
  'turbo210': ('2.1.0',210)
}

# requirements
try:
  with open('requirements.txt') as f:
    reqs = f.read().splitlines()
except:
  reqs = ['numpy']

import codecs
import setuptools
with codecs.open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

# create version dependent extensions
import ctypes
from pathlib import Path
import re
cfiles = {}
hfiles = {}
cjpeglib = {}
for v in libjpeg_versions:
  is_turbo = v[:5] == "turbo"
  clib = f'jpeglib/cjpeglib/{v}'
  print(clib)
  
  package_name = 'libjpeg'
  (Path(clib) / 'jconfig.h').touch()
  if is_turbo:
    package_name += '-turbo'
    (Path(clib) / 'jconfigint.h').touch()

  files = [f'{clib}/{f}' for f in os.listdir(clib) if re.fullmatch(f'.*\.(c|h)', f)]
  for excluded_module in ['jmemdos','jmemmac','jmemansi','ansi2knr','ckconfig','jmemname', # platform dependents
                          'djpeg','cjpeg','rdjpgcom','wrjpgcom','cdjpeg','jpegtran', # executables
                          'rdbmp','wrbmp','rdcolmap','rdppm','wrppm','rdtarga','wrtarga','rdrle','wrrle','rdgif','wrgif','rdswitch', # others
                          'example', # example
                          #'jerror',
                          # turbo
                          'jccolext','jdcolext','jdcol565','jdmrg565','jdmrgext',"jcstest","tjunittest","tjbench",
                          'jstdhuff','turbojpeg-jni','turbojpeg']: 
    lim = -2 - len(excluded_module)
    files = [f for f in files if f[lim:-2] != excluded_module]
  #
  cfiles[v] = [f for f in files if f[-2:] == '.c']
  hfiles[v] = [f for f in files if f[-2:] == '.h']
  sources = ['jpeglib/cjpeglib/cjpeglib.c',*cfiles[v]]
  
  turbo_macros = [
    ("JPEG_LIB_VERSION",70),
    ("INLINE","__inline__"),
    ("PACKAGE_NAME",f"\"{package_name}\""),
    ("BUILD",f"\"unknown\""),
    ("VERSION",f"\"{libjpeg_versions[v][0]}\""),
    ("SIZEOF_SIZE_T",int(ctypes.sizeof(ctypes.c_size_t))),
    ("THREAD_LOCAL", "__thread")
  ] if is_turbo else []

  cjpeglib[v] = setuptools.Extension(
    name = f"jpeglib/cjpeglib/cjpeglib_{v}",
    library_dirs=['./jpeglib/cjpeglib',f'./{clib}'],#[f'./{clib}'],
    include_dirs=['./jpeglib/cjpeglib',f'./{clib}'],#[f'./{clib}'],
    sources = sources,
    headers = hfiles[v],
    define_macros = [
      ("BITS_IN_JSAMPLE",8),
      ("HAVE_STDLIB_H",1),
      ("LIBVERSION",libjpeg_versions[v][1]),
      ("HAVE_PROTOTYPES",1),
      *turbo_macros
    ],
    extra_compile_args=["-fPIC","-g"],
    language="c",
  )

setuptools.setup(
  name = 'jpeglib',
  version = __version__,
  author = u'Martin Beneš',
  author_email = 'martinbenes1996@gmail.com',
  description = 'Python envelope for the popular C library libjpeg for handling JPEG files.',
  long_description = long_description,
  long_description_content_type="text/markdown",
  packages=setuptools.find_packages(),
  license='MPL',
  #test_suite = 'setup.test_suite',
  url = 'https://jpeglib.readthedocs.io/en/latest/',
  #download_url = 'https://github.com/martinbenes1996/jpeglib/archive/0.1.0.tar.gz',
  keywords = ['jpeglib','jpeg','jpg','libjpeg','compression','decompression','dct-coefficients','dct'],
  install_requires=reqs,
  package_dir={'': '.'},
  package_data={'': ['data/*']},
  include_package_data=True,
  ext_modules=[cjpeglib[v] for v in libjpeg_versions],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Information Technology',
    'Intended Audience :: Other Audience',
    'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: POSIX :: Linux',
    'Programming Language :: C',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9',
    'Topic :: Education',
    'Topic :: Multimedia',
    'Topic :: Multimedia :: Graphics :: Graphics Conversion',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Image Processing',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'Topic :: Scientific/Engineering :: Information Analysis',
    'Topic :: Security',
    'Topic :: Software Development :: Libraries',
    'Topic :: Software Development :: Libraries :: Python Modules',
    'Topic :: Utilities'
  ],
)
