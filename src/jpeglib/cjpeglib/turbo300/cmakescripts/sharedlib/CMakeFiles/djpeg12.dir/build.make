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
include sharedlib/CMakeFiles/djpeg12.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include sharedlib/CMakeFiles/djpeg12.dir/compiler_depend.make

# Include the progress variables for this target.
include sharedlib/CMakeFiles/djpeg12.dir/progress.make

# Include the compile flags for this target's objects.
include sharedlib/CMakeFiles/djpeg12.dir/flags.make

sharedlib/CMakeFiles/djpeg12.dir/__/rdcolmap.c.o: sharedlib/CMakeFiles/djpeg12.dir/flags.make
sharedlib/CMakeFiles/djpeg12.dir/__/rdcolmap.c.o: /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/rdcolmap.c
sharedlib/CMakeFiles/djpeg12.dir/__/rdcolmap.c.o: sharedlib/CMakeFiles/djpeg12.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object sharedlib/CMakeFiles/djpeg12.dir/__/rdcolmap.c.o"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/sharedlib && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT sharedlib/CMakeFiles/djpeg12.dir/__/rdcolmap.c.o -MF CMakeFiles/djpeg12.dir/__/rdcolmap.c.o.d -o CMakeFiles/djpeg12.dir/__/rdcolmap.c.o -c /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/rdcolmap.c

sharedlib/CMakeFiles/djpeg12.dir/__/rdcolmap.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/djpeg12.dir/__/rdcolmap.c.i"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/sharedlib && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/rdcolmap.c > CMakeFiles/djpeg12.dir/__/rdcolmap.c.i

sharedlib/CMakeFiles/djpeg12.dir/__/rdcolmap.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/djpeg12.dir/__/rdcolmap.c.s"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/sharedlib && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/rdcolmap.c -o CMakeFiles/djpeg12.dir/__/rdcolmap.c.s

sharedlib/CMakeFiles/djpeg12.dir/__/wrgif.c.o: sharedlib/CMakeFiles/djpeg12.dir/flags.make
sharedlib/CMakeFiles/djpeg12.dir/__/wrgif.c.o: /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/wrgif.c
sharedlib/CMakeFiles/djpeg12.dir/__/wrgif.c.o: sharedlib/CMakeFiles/djpeg12.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object sharedlib/CMakeFiles/djpeg12.dir/__/wrgif.c.o"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/sharedlib && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT sharedlib/CMakeFiles/djpeg12.dir/__/wrgif.c.o -MF CMakeFiles/djpeg12.dir/__/wrgif.c.o.d -o CMakeFiles/djpeg12.dir/__/wrgif.c.o -c /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/wrgif.c

sharedlib/CMakeFiles/djpeg12.dir/__/wrgif.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/djpeg12.dir/__/wrgif.c.i"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/sharedlib && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/wrgif.c > CMakeFiles/djpeg12.dir/__/wrgif.c.i

sharedlib/CMakeFiles/djpeg12.dir/__/wrgif.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/djpeg12.dir/__/wrgif.c.s"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/sharedlib && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/wrgif.c -o CMakeFiles/djpeg12.dir/__/wrgif.c.s

sharedlib/CMakeFiles/djpeg12.dir/__/wrppm.c.o: sharedlib/CMakeFiles/djpeg12.dir/flags.make
sharedlib/CMakeFiles/djpeg12.dir/__/wrppm.c.o: /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/wrppm.c
sharedlib/CMakeFiles/djpeg12.dir/__/wrppm.c.o: sharedlib/CMakeFiles/djpeg12.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object sharedlib/CMakeFiles/djpeg12.dir/__/wrppm.c.o"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/sharedlib && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT sharedlib/CMakeFiles/djpeg12.dir/__/wrppm.c.o -MF CMakeFiles/djpeg12.dir/__/wrppm.c.o.d -o CMakeFiles/djpeg12.dir/__/wrppm.c.o -c /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/wrppm.c

sharedlib/CMakeFiles/djpeg12.dir/__/wrppm.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/djpeg12.dir/__/wrppm.c.i"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/sharedlib && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/wrppm.c > CMakeFiles/djpeg12.dir/__/wrppm.c.i

sharedlib/CMakeFiles/djpeg12.dir/__/wrppm.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/djpeg12.dir/__/wrppm.c.s"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/sharedlib && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/wrppm.c -o CMakeFiles/djpeg12.dir/__/wrppm.c.s

djpeg12: sharedlib/CMakeFiles/djpeg12.dir/__/rdcolmap.c.o
djpeg12: sharedlib/CMakeFiles/djpeg12.dir/__/wrgif.c.o
djpeg12: sharedlib/CMakeFiles/djpeg12.dir/__/wrppm.c.o
djpeg12: sharedlib/CMakeFiles/djpeg12.dir/build.make
.PHONY : djpeg12

# Rule to build all files generated by this target.
sharedlib/CMakeFiles/djpeg12.dir/build: djpeg12
.PHONY : sharedlib/CMakeFiles/djpeg12.dir/build

sharedlib/CMakeFiles/djpeg12.dir/clean:
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/sharedlib && $(CMAKE_COMMAND) -P CMakeFiles/djpeg12.dir/cmake_clean.cmake
.PHONY : sharedlib/CMakeFiles/djpeg12.dir/clean

sharedlib/CMakeFiles/djpeg12.dir/depend:
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300 /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/sharedlib /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/sharedlib /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/sharedlib/CMakeFiles/djpeg12.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : sharedlib/CMakeFiles/djpeg12.dir/depend

