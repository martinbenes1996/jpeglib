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
include md5/CMakeFiles/md5cmp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include md5/CMakeFiles/md5cmp.dir/compiler_depend.make

# Include the progress variables for this target.
include md5/CMakeFiles/md5cmp.dir/progress.make

# Include the compile flags for this target's objects.
include md5/CMakeFiles/md5cmp.dir/flags.make

md5/CMakeFiles/md5cmp.dir/md5cmp.c.o: md5/CMakeFiles/md5cmp.dir/flags.make
md5/CMakeFiles/md5cmp.dir/md5cmp.c.o: /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/md5/md5cmp.c
md5/CMakeFiles/md5cmp.dir/md5cmp.c.o: md5/CMakeFiles/md5cmp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object md5/CMakeFiles/md5cmp.dir/md5cmp.c.o"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/md5 && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT md5/CMakeFiles/md5cmp.dir/md5cmp.c.o -MF CMakeFiles/md5cmp.dir/md5cmp.c.o.d -o CMakeFiles/md5cmp.dir/md5cmp.c.o -c /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/md5/md5cmp.c

md5/CMakeFiles/md5cmp.dir/md5cmp.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/md5cmp.dir/md5cmp.c.i"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/md5 && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/md5/md5cmp.c > CMakeFiles/md5cmp.dir/md5cmp.c.i

md5/CMakeFiles/md5cmp.dir/md5cmp.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/md5cmp.dir/md5cmp.c.s"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/md5 && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/md5/md5cmp.c -o CMakeFiles/md5cmp.dir/md5cmp.c.s

md5/CMakeFiles/md5cmp.dir/md5.c.o: md5/CMakeFiles/md5cmp.dir/flags.make
md5/CMakeFiles/md5cmp.dir/md5.c.o: /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/md5/md5.c
md5/CMakeFiles/md5cmp.dir/md5.c.o: md5/CMakeFiles/md5cmp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object md5/CMakeFiles/md5cmp.dir/md5.c.o"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/md5 && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT md5/CMakeFiles/md5cmp.dir/md5.c.o -MF CMakeFiles/md5cmp.dir/md5.c.o.d -o CMakeFiles/md5cmp.dir/md5.c.o -c /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/md5/md5.c

md5/CMakeFiles/md5cmp.dir/md5.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/md5cmp.dir/md5.c.i"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/md5 && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/md5/md5.c > CMakeFiles/md5cmp.dir/md5.c.i

md5/CMakeFiles/md5cmp.dir/md5.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/md5cmp.dir/md5.c.s"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/md5 && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/md5/md5.c -o CMakeFiles/md5cmp.dir/md5.c.s

md5/CMakeFiles/md5cmp.dir/md5hl.c.o: md5/CMakeFiles/md5cmp.dir/flags.make
md5/CMakeFiles/md5cmp.dir/md5hl.c.o: /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/md5/md5hl.c
md5/CMakeFiles/md5cmp.dir/md5hl.c.o: md5/CMakeFiles/md5cmp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object md5/CMakeFiles/md5cmp.dir/md5hl.c.o"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/md5 && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT md5/CMakeFiles/md5cmp.dir/md5hl.c.o -MF CMakeFiles/md5cmp.dir/md5hl.c.o.d -o CMakeFiles/md5cmp.dir/md5hl.c.o -c /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/md5/md5hl.c

md5/CMakeFiles/md5cmp.dir/md5hl.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/md5cmp.dir/md5hl.c.i"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/md5 && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/md5/md5hl.c > CMakeFiles/md5cmp.dir/md5hl.c.i

md5/CMakeFiles/md5cmp.dir/md5hl.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/md5cmp.dir/md5hl.c.s"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/md5 && /Library/Developer/CommandLineTools/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/md5/md5hl.c -o CMakeFiles/md5cmp.dir/md5hl.c.s

# Object files for target md5cmp
md5cmp_OBJECTS = \
"CMakeFiles/md5cmp.dir/md5cmp.c.o" \
"CMakeFiles/md5cmp.dir/md5.c.o" \
"CMakeFiles/md5cmp.dir/md5hl.c.o"

# External object files for target md5cmp
md5cmp_EXTERNAL_OBJECTS =

md5/md5cmp: md5/CMakeFiles/md5cmp.dir/md5cmp.c.o
md5/md5cmp: md5/CMakeFiles/md5cmp.dir/md5.c.o
md5/md5cmp: md5/CMakeFiles/md5cmp.dir/md5hl.c.o
md5/md5cmp: md5/CMakeFiles/md5cmp.dir/build.make
md5/md5cmp: md5/CMakeFiles/md5cmp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C executable md5cmp"
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/md5 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/md5cmp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
md5/CMakeFiles/md5cmp.dir/build: md5/md5cmp
.PHONY : md5/CMakeFiles/md5cmp.dir/build

md5/CMakeFiles/md5cmp.dir/clean:
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/md5 && $(CMAKE_COMMAND) -P CMakeFiles/md5cmp.dir/cmake_clean.cmake
.PHONY : md5/CMakeFiles/md5cmp.dir/clean

md5/CMakeFiles/md5cmp.dir/depend:
	cd /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300 /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/md5 /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/md5 /Users/martin/UIBK/jpeglib/jpeglib/cjpeglib/turbo300/cmakescripts/md5/CMakeFiles/md5cmp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : md5/CMakeFiles/md5cmp.dir/depend
