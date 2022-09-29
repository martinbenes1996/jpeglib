

import codecs
import ctypes
import os
from pathlib import Path
import re
import setuptools
import setuptools.command.build_ext
import sys

# wheel builder
try:
    from wheel.bdist_wheel import bdist_wheel

    class bdist_wheel_abi3(bdist_wheel):
        def get_tag(self):
            python, abi, plat = super().get_tag()
            if python.startswith("cp"):
                return "cp38", "abi3", plat
            return python, abi, plat

    custom_bdist_wheel = {'bdist_wheel': bdist_wheel_abi3}
except ModuleNotFoundError:
    custom_bdist_wheel = {}

# versions
__version__ = os.environ.get('VERSION_NEW', '0.11.2')
libjpeg_versions = {
    '6b': (None, 60),
    '7': (None, 70),
    '8': (None, 80),
    '8a': (None, 80),
    '8b': (None, 80),
    '8c': (None, 80),
    '8d': (None, 80),
    '9': (None, 90),
    '9a': (None, 90),
    '9b': (None, 90),
    '9c': (None, 90),
    '9d': (None, 90),
    '9e': (None, 90),
    'turbo120': ('1.2.0', 120),
    'turbo130': ('1.3.0', 130),
    'turbo140': ('1.4.0', 140),
    'turbo150': ('1.5.0', 150),
    'turbo200': ('2.0.0', 200),
    'turbo210': ('2.1.0', 210),
    'mozjpeg101': ('1.0.1', 101),
    'mozjpeg201': ('2.0.1', 201),
    'mozjpeg300': ('3.0.0', 300),
    'mozjpeg403': ('4.0.3', 403),
}

# requirements
try:
    with open('requirements.txt') as f:
        reqs = f.read().splitlines()
except FileNotFoundError:
    reqs = ['numpy']

# description
with codecs.open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

# create version dependent extensions
cfiles, hfiles = {}, {}
cjpeglib = {}
for v in libjpeg_versions:

    # library-dependent
    is_moz = v[:3] == "moz"
    is_turbo_moz = v[:5] == "turbo" or is_moz

    # name of C library
    clib = f'jpeglib/cjpeglib/{v}'

    # create missing
    package_name = 'libjpeg'
    (Path(clib) / 'jconfig.h').touch()
    (Path(clib) / 'config.h').touch()
    if not (Path(clib) / 'vjpeglib.h').exists():
        with open(Path(clib) / 'vjpeglib.h','w') as f:
            f.write('\n#include "jpeglib.h"\n')
    if is_turbo_moz:
        package_name += '-turbo'
        (Path(clib) / 'jconfigint.h').touch()
        if is_moz:
            package_name = 'mozjpeg'
            (Path(clib) / 'config.h').touch()

    # get all files
    files = [
        f'{clib}/{f}'
        for f in os.listdir(clib)
        if re.fullmatch(r'.*\.(c|h)', f)
    ]
    # exclude files
    for excluded_module in [
        # platform dependents
        'jmemdos', 'jmemmac', 'jmemansi',
        'ansi2knr', 'ckconfig', 'jmemname',
        # executables
        'djpeg', 'cjpeg', 'rdjpgcom',
        'wrjpgcom', 'cdjpeg', 'jpegtran',
        # others
        'rdbmp', 'wrbmp', 'rdcolmap',
        'rdppm', 'wrppm', 'rdtarga',
        'wrtarga', 'rdrle', 'wrrle',
        'rdgif', 'wrgif', 'rdswitch',
        # example
        'example',
        #
        'cjpegalt', 'djpegalt',
        # turbo
        'jccolext', 'jdcolext', 'jdcol565',
        'jstdhuff', 'jdmrg565', 'jdmrgext',
        'jcstest', 'tjunittest', 'tjbench',
        'turbojpeg-jni', 'turbojpeg', 'turbojpegl',
        'jpegut', 'jpgtest',
        # mozjpeg
        'bmp', 'jpegyuv',
    ]:
        lim = -2 - len(excluded_module)
        files = [f for f in files if f[lim:-2] != excluded_module]
    # split to sources and headers
    cfiles[v] = [f for f in files if f[-2:] == '.c']
    hfiles[v] = [f for f in files if f[-2:] == '.h']
    sources = ['jpeglib/cjpeglib/cjpeglib.c', *cfiles[v]]

    # define macros
    macros = [
        ("BITS_IN_JSAMPLE", 8),
        ("HAVE_STDLIB_H", 1),
        ("LIBVERSION", libjpeg_versions[v][1]),
        ("HAVE_PROTOTYPES", 1),
        ("Py_LIMITED_API", "0x03080000"),
    ]
    # turbo/moz-only macros
    if is_turbo_moz:
        macros += [
            ("INLINE", "__inline__" if not sys.platform.startswith("win") else "__inline"),
            ("PACKAGE_NAME", f"\"{package_name}\""),
            ("BUILD", "\"unknown\""),
            ("VERSION", f"\"{libjpeg_versions[v][0]}\""),
            ("SIZEOF_SIZE_T", int(ctypes.sizeof(ctypes.c_size_t))),
            ("THREAD_LOCAL", "__thread"),
            ("JPEG_LIB_VERSION", 70),  # 70), # turbo 2.1.0
            ("C_ARITH_CODING_SUPPORTED", 1),
            ("D_ARITH_CODING_SUPPORTED", 1),
        ]
        # moz-only macros
        if is_moz:
            macros += [
                ("JPEG_LIB_VERSION", 69),
                ('MEM_SRCDST_SUPPORTED', 1)
            ]

    # define the extension
    cjpeglib[v] = setuptools.Extension(
        name=f"jpeglib/cjpeglib/cjpeglib_{v}",
        library_dirs=['./jpeglib/cjpeglib', f'./{clib}'],
        include_dirs=['./jpeglib/cjpeglib', f'./{clib}'],
        sources=sources,
        headers=hfiles[v],
        define_macros=macros,
        extra_compile_args=["-fPIC", "-g"],
        language="C",
        py_limited_api=True,
    )

# extension builder
class custom_build_ext(setuptools.command.build_ext.build_ext):
    def get_export_symbols(self, ext):
        parts = ext.name.split(".")
        if parts[-1] == "__init__":
            initfunc_name = "PyInit_" + parts[-2]
        else:
            initfunc_name = "PyInit_" + parts[-1]

    def build_extensions(self):
        setuptools.command.build_ext.build_ext.build_extensions(self)
        setuptools.command.build_ext.build_ext.get_export_symbols = self.get_export_symbols

# define package
setuptools.setup(
    name='jpeglib',
    version=__version__,
    author=u'Martin Beneš',
    author_email='martinbenes1996@gmail.com',
    description="Python envelope for the popular C library" +
                "libjpeg for handling JPEG files.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    license='MPL',
    project_urls={
        "Homepage": "https://pypi.org/project/jpeglib/",
        "Documentation": 'https://jpeglib.readthedocs.io/en/latest/',
        "Source": "https://github.com/martinbenes1996/jpeglib/",
    },
    keywords=['jpeglib', 'jpeg', 'jpg', 'libjpeg', 'compression',
              'decompression', 'dct-coefficients', 'dct'],
    install_requires=reqs,
    package_dir={'': '.'},
    package_data={'': ['data/*']},
    include_package_data=True,
    ext_modules=[cjpeglib[v] for v in libjpeg_versions],
    cmdclass={"build_ext": custom_build_ext, **custom_bdist_wheel},
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
        'Topic :: Education',
        'Topic :: Multimedia',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Multimedia :: Graphics :: Graphics Conversion',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Security',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Utilities',
    ],
)
