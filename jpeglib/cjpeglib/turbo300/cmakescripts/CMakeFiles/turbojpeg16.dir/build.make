# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.25.3/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.25.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts

# Include any dependencies generated for this target.
include CMakeFiles/turbojpeg16.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/turbojpeg16.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/turbojpeg16.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/turbojpeg16.dir/flags.make

CMakeFiles/turbojpeg16.dir/rdppm.c.o: CMakeFiles/turbojpeg16.dir/flags.make
CMakeFiles/turbojpeg16.dir/rdppm.c.o: /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/rdppm.c
CMakeFiles/turbojpeg16.dir/rdppm.c.o: CMakeFiles/turbojpeg16.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/turbojpeg16.dir/rdppm.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/turbojpeg16.dir/rdppm.c.o -MF CMakeFiles/turbojpeg16.dir/rdppm.c.o.d -o CMakeFiles/turbojpeg16.dir/rdppm.c.o -c /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/rdppm.c

CMakeFiles/turbojpeg16.dir/rdppm.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/turbojpeg16.dir/rdppm.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/rdppm.c > CMakeFiles/turbojpeg16.dir/rdppm.c.i

CMakeFiles/turbojpeg16.dir/rdppm.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/turbojpeg16.dir/rdppm.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/rdppm.c -o CMakeFiles/turbojpeg16.dir/rdppm.c.s

CMakeFiles/turbojpeg16.dir/wrppm.c.o: CMakeFiles/turbojpeg16.dir/flags.make
CMakeFiles/turbojpeg16.dir/wrppm.c.o: /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/wrppm.c
CMakeFiles/turbojpeg16.dir/wrppm.c.o: CMakeFiles/turbojpeg16.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/turbojpeg16.dir/wrppm.c.o"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/turbojpeg16.dir/wrppm.c.o -MF CMakeFiles/turbojpeg16.dir/wrppm.c.o.d -o CMakeFiles/turbojpeg16.dir/wrppm.c.o -c /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/wrppm.c

CMakeFiles/turbojpeg16.dir/wrppm.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/turbojpeg16.dir/wrppm.c.i"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/wrppm.c > CMakeFiles/turbojpeg16.dir/wrppm.c.i

CMakeFiles/turbojpeg16.dir/wrppm.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/turbojpeg16.dir/wrppm.c.s"
	/Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/wrppm.c -o CMakeFiles/turbojpeg16.dir/wrppm.c.s

turbojpeg16: CMakeFiles/turbojpeg16.dir/rdppm.c.o
turbojpeg16: CMakeFiles/turbojpeg16.dir/wrppm.c.o
turbojpeg16: CMakeFiles/turbojpeg16.dir/build.make
.PHONY : turbojpeg16

# Rule to build all files generated by this target.
CMakeFiles/turbojpeg16.dir/build: turbojpeg16
.PHONY : CMakeFiles/turbojpeg16.dir/build

CMakeFiles/turbojpeg16.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/turbojpeg16.dir/cmake_clean.cmake
.PHONY : CMakeFiles/turbojpeg16.dir/clean

CMakeFiles/turbojpeg16.dir/depend:
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300 /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300 /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/CMakeFiles/turbojpeg16.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/turbojpeg16.dir/depend

