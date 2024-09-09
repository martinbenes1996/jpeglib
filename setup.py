"""

Author: Martin Benes
Affiliation: University of Innsbruck
"""

import codecs
import ctypes
import glob
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
            if python.startswith('cp'):
                return 'cp38', 'abi3', plat
            return python, abi, plat

    custom_bdist_wheel = {'bdist_wheel': bdist_wheel_abi3}
except ModuleNotFoundError:
    custom_bdist_wheel = {}

# versions
__version__ = os.environ.get('VERSION_NEW', '1.0.1')
libjpeg_versions = {
    '6b': (None, 60),
    '7': (None, 70),
    '8': (None, 80),
    '8a': (None, 81),
    '8b': (None, 82),
    '8c': (None, 83),
    '8d': (None, 84),
    '9': (None, 90),
    '9a': (None, 91),
    '9b': (None, 92),
    '9c': (None, 93),
    '9d': (None, 94),
    '9e': (None, 95),
    '9f': (None, 96),
    'turbo120': ('1.2.0', 3120),
    'turbo130': ('1.3.0', 3130),
    'turbo140': ('1.4.0', 3140),
    'turbo150': ('1.5.0', 3150),
    'turbo200': ('2.0.0', 3200),
    'turbo210': ('2.1.0', 3210),
    # 'turbo300': ('3.0.0', 3300),
    'mozjpeg101': ('1.0.1', 6101),
    'mozjpeg201': ('2.0.1', 6201),
    'mozjpeg300': ('3.0.0', 6300),
    'mozjpeg403': ('4.0.3', 6403),
}

# requirements
with open('requirements.txt') as f:
    reqs = f.read().splitlines()

# description
with codecs.open('README.md', 'r', encoding='UTF-8') as fh:
    long_description = fh.read()

# create version dependent extensions
cfiles, hfiles = {}, {}
cjpeglib = {}
for v in libjpeg_versions:

    # library-dependent
    is_moz = v[:3] == 'moz'
    is_turbo = v[:5] == 'turbo'

    # name of C library
    rootlib = Path('src') / 'jpeglib' / 'cjpeglib'
    clib = rootlib / v

    # create missing
    package_name = 'libjpeg'
    (clib / 'jconfig.h').touch()
    (clib / 'config.h').touch()
    if True:  # not (Path(clib) / 'vjpeglib.h').exists():
        with open(clib / 'vjpeglib.h', 'w') as f:
            f.write('#include "jpeglib.h"')
    if is_turbo or is_moz:
        package_name += '-turbo'
        (clib / 'jconfigint.h').touch()
        if is_moz:
            package_name = 'mozjpeg'
            (clib / 'config.h').touch()
        else:
            (clib / 'jversion.h').touch()

    #
    simddirs = []
    # if is_turbo:
    #     simddirs.append(clib/'simd')
    #     # simddirs.append(clib/'simd'/'arm')
    #     simddirs.append(clib/'simd'/'i386')
    #     # simddirs.append(clib/'simd'/'mips')
    #     # simddirs.append(clib/'simd'/'mips64')
    #     # simddirs.append(clib/'simd'/'nasm')
    #     # simddirs.append(clib/'simd'/'powerpc')
    #     simddirs.append(clib/'simd'/'x86_64')

    # get all files
    files = [
        f
        for d in [rootlib, clib, *simddirs]
        for f in glob.glob(str(d/'*'))
        if re.fullmatch(r'.*\.(c|h|cpp)', f)
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
        'turbojpeg-mp',
        'jsimd',
        'jsimddct',
        'jcgryext-neon',
        'jdcolext-neon',
        'jdmrgext-neon',
        # mozjpeg
        'bmp', 'jpegyuv',
    ]:
        lim = -2 - len(excluded_module)
        files = [f for f in files if f[lim:-2] != excluded_module]

    # split to sources and headers
    cfiles[v] = [
        f for f in files
        if any(f.endswith(s) for s in ['c', '.cpp'])
    ]
    hfiles[v] = [
        f for f in files
        if f[-2:] == '.h'
    ]
    sources = cfiles[v]

    # define macros
    macros = [
        ('BITS_IN_JSAMPLE', 8),
        ('HAVE_UNSIGNED_CHAR', 1),
        ('HAVE_STDLIB_H', 1),
        ('LIBVERSION', libjpeg_versions[v][1]),
        ('HAVE_PROTOTYPES', 1),
        ('Py_LIMITED_API', '0x03080000'),
        # ('C_LOSSLESS_SUPPORTED', 1),
        # ('D_LOSSLESS_SUPPORTED', 1),
        # ('WITH_SIMD', 0),  # TODO: 1
    ]
    # turbo/moz-only macros
    if is_turbo or is_moz:
        macros += [
            ('INLINE', '__inline__' if not sys.platform.startswith('win') else '__inline'),
            ('PACKAGE_NAME', f'"{package_name}"'),
            ('BUILD', '"unknown"'),
            ('VERSION', f'"{libjpeg_versions[v][0]}"'),
            ('SIZEOF_SIZE_T', int(ctypes.sizeof(ctypes.c_size_t))),
            ('THREAD_LOCAL', '__thread'),
            ('C_ARITH_CODING_SUPPORTED', 1),
            ('D_ARITH_CODING_SUPPORTED', 1),
            ('JPEG_LIB_VERSION', 70),
        ]
        # moz-only macros
        if is_moz:
            macros += [
                # ('JPEG_LIB_VERSION', 69),
                ('MEM_SRCDST_SUPPORTED', 1),
            ]
        else:
            macros += [
                ('FALLTHROUGH', ''),
            ]

    # define the extension
    cjpeglib[v] = setuptools.Extension(
        name=f'jpeglib.cjpeglib.cjpeglib_{v}',
        library_dirs=['jpeglib/cjpeglib', f'{clib}'],
        include_dirs=['jpeglib/cjpeglib', f'{clib}'],
        sources=sources,
        headers=hfiles[v],
        define_macros=macros,
        extra_compile_args=[] if sys.platform.startswith('win') else ['-fPIC'],
        # language='C++',
        py_limited_api=True,
    )


# extension builder
class custom_build_ext(setuptools.command.build_ext.build_ext):
    def get_export_symbols(self, ext):
        parts = ext.name.split('.')
        if parts[-1] == '__init__':
            initfunc_name = 'PyInit_' + parts[-2]
        else:
            initfunc_name = 'PyInit_' + parts[-1]

    def build_extensions(self):
        setuptools.command.build_ext.build_ext.build_extensions(self)
        setuptools.command.build_ext.build_ext.get_export_symbols = self.get_export_symbols


# define package
setuptools.setup(
    name='jpeglib',
    version=__version__,
    author=u'Martin Beneš',
    author_email='martinbenes1996@gmail.com',
    description='Python envelope for the popular C library ' +
                'libjpeg for handling JPEG files.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['jpeglib'],
    license='MPL',
    project_urls={
        'Homepage': 'https://pypi.org/project/jpeglib/',
        'Documentation': 'https://jpeglib.readthedocs.io/en/latest/',
        'Source': 'https://github.com/martinbenes1996/jpeglib/',
    },
    keywords=['jpeglib', 'jpeg', 'jpg', 'libjpeg', 'compression',
              'decompression', 'dct-coefficients', 'dct'],
    install_requires=reqs,
    package_dir={'': 'src'},
    package_data={'': ['data/*']},
    include_package_data=True,
    ext_modules=[cjpeglib[v] for v in libjpeg_versions],
    cmdclass={
        'build_ext': custom_build_ext,
        **custom_bdist_wheel
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
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
