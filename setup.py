
# versions
from setuptools.command.build_ext import build_ext
import re
from pathlib import Path
import ctypes
import setuptools
import codecs
import os
__version__ = os.environ.get('VERSION_NEW', '0.10.11')
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
    'turbo210': ('2.1.0', 210),
    'mozjpeg403': ('4.0.3', 403)
}

# requirements
try:
    with open('requirements.txt') as f:
        reqs = f.read().splitlines()
except:
    reqs = ['numpy']

with codecs.open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

# create version dependent extensions
cfiles = {}
hfiles = {}
cjpeglib = {}
for v in libjpeg_versions:
    is_moz = v[:3] == "moz"
    is_turbo = v[:5] == "turbo" or is_moz

    clib = f'jpeglib/cjpeglib/{v}'

    # create missing
    package_name = 'libjpeg'
    (Path(clib) / 'jconfig.h').touch()
    (Path(clib) / 'vjpeglib.h').touch()
    if is_turbo:
        package_name += '-turbo'
        (Path(clib) / 'jconfigint.h').touch()

    files = [
        f'{clib}/{f}' for f in os.listdir(clib) if re.fullmatch(f'.*\.(c|h)', f)]
    for excluded_module in ['jmemdos', 'jmemmac', 'jmemansi', 'ansi2knr', 'ckconfig', 'jmemname',  # platform dependents
                            'djpeg', 'cjpeg', 'rdjpgcom', 'wrjpgcom', 'cdjpeg', 'jpegtran',  # executables
                            'rdbmp', 'wrbmp', 'rdcolmap', 'rdppm', 'wrppm', 'rdtarga', 'wrtarga', 'rdrle', 'wrrle', 'rdgif', 'wrgif', 'rdswitch',  # others
                            'example',  # example
                            'cjpegalt', 'djpegalt',
                            # 'jerror',
                            # turbo
                            'jccolext', 'jdcolext', 'jdcol565', 'jstdhuff',
                            'jdmrg565', 'jdmrgext', "jcstest", "tjunittest", "tjbench",
                            'turbojpeg-jni', 'turbojpeg']:
        lim = -2 - len(excluded_module)
        files = [f for f in files if f[lim:-2] != excluded_module]
    #
    cfiles[v] = [f for f in files if f[-2:] == '.c']
    hfiles[v] = [f for f in files if f[-2:] == '.h']
    sources = ['jpeglib/cjpeglib/cjpeglib.c', *cfiles[v]]

    macros = [
        ("BITS_IN_JSAMPLE", 8),
        ("HAVE_STDLIB_H", 1),
        ("LIBVERSION", libjpeg_versions[v][1]),
        ("HAVE_PROTOTYPES", 1),
    ]
    if is_turbo:
        macros += [
            ("JPEG_LIB_VERSION", 80),  # 70),
            ("INLINE", "__inline__"),
            ("PACKAGE_NAME", f"\"{package_name}\""),
            ("BUILD", f"\"unknown\""),
            ("VERSION", f"\"{libjpeg_versions[v][0]}\""),
            ("SIZEOF_SIZE_T", int(ctypes.sizeof(ctypes.c_size_t))),
            ("THREAD_LOCAL", "__thread")
        ]
    if is_moz:
        macros += [
            ('C_ARITH_CODING_SUPPORTED', 1)
        ]

    cjpeglib[v] = setuptools.Extension(
        name=f"jpeglib/cjpeglib/cjpeglib_{v}",
        library_dirs=['./jpeglib/cjpeglib', f'./{clib}'],  # [f'./{clib}'],
        include_dirs=['./jpeglib/cjpeglib', f'./{clib}'],  # [f'./{clib}'],
        sources=sources,
        headers=hfiles[v],
        define_macros=macros,
        extra_compile_args=["-fPIC", "-g"],
        language="c",
    )


class custom_build_ext(build_ext):
    def build_extensions(self):
        #self.compiler.set_executable("compiler_so", "g++")
        #self.compiler.set_executable("compiler_cxx", "g++")
        #self.compiler.set_executable("linker_so", "g++")
        # 'add_include_dir', 'add_library', 'add_library_dir', 'add_link_object', 'add_runtime_library_dir',
        # 'announce', 'archiver', 'compile', 'compiler', 'compiler_cxx', 'compiler_so', 'compiler_type',
        # 'create_static_lib', 'debug_print', 'define_macro', 'detect_language', 'dry_run', 'dylib_lib_extension',
        # 'dylib_lib_format', 'exe_extension', 'executable_filename', 'executables', 'execute', 'find_library_file',
        # 'force', 'has_function', 'include_dirs', 'language_map', 'language_order', 'libraries', 'library_dir_option',
        # 'library_dirs', 'library_filename', 'library_option', 'link', 'link_executable', 'link_shared_lib',
        # 'link_shared_object', 'linker_exe', 'linker_so', 'macros', 'mkpath', 'move_file', 'obj_extension',
        # 'object_filenames', 'objects', 'output_dir', 'preprocess', 'preprocessor', 'ranlib', 'runtime_library_dir_option',
        # 'runtime_library_dirs', 'set_executable', 'set_executables', 'set_include_dirs', 'set_libraries', 'set_library_dirs',
        # 'set_link_objects', 'set_runtime_library_dirs', 'shared_lib_extension', 'shared_lib_format', 'shared_object_filename',
        # 'spawn', 'src_extensions', 'static_lib_extension', 'static_lib_format', 'undefine_macro', 'verbose', 'warn',
        # 'xcode_stub_lib_extension', 'xcode_stub_lib_format'
        #print("==========", self.compiler.library_dirs)
        build_ext.build_extensions(self)


setuptools.setup(
    name='jpeglib',
    version=__version__,
    author=u'Martin Beneš',
    author_email='martinbenes1996@gmail.com',
    description='Python envelope for the popular C library libjpeg for handling JPEG files.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    license='MPL',
    #test_suite = 'setup.test_suite',
    url='https://jpeglib.readthedocs.io/en/latest/',
    #download_url = 'https://github.com/martinbenes1996/jpeglib/archive/0.1.0.tar.gz',
    keywords=['jpeglib', 'jpeg', 'jpg', 'libjpeg', 'compression',
              'decompression', 'dct-coefficients', 'dct'],
    install_requires=reqs,
    package_dir={'': '.'},
    package_data={'': ['data/*']},
    include_package_data=True,
    ext_modules=[cjpeglib[v] for v in libjpeg_versions],
    cmdclass={"build_ext": custom_build_ext},
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
