# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.16.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.16.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/xuzhiyuan/CLionProjects/xflow

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/xuzhiyuan/CLionProjects/xflow

# Include any dependencies generated for this target.
include src/io/CMakeFiles/io.dir/depend.make

# Include the progress variables for this target.
include src/io/CMakeFiles/io.dir/progress.make

# Include the compile flags for this target's objects.
include src/io/CMakeFiles/io.dir/flags.make

src/io/CMakeFiles/io.dir/load_data_from_disk.cc.o: src/io/CMakeFiles/io.dir/flags.make
src/io/CMakeFiles/io.dir/load_data_from_disk.cc.o: src/io/load_data_from_disk.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xuzhiyuan/CLionProjects/xflow/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/io/CMakeFiles/io.dir/load_data_from_disk.cc.o"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/io && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/io.dir/load_data_from_disk.cc.o -c /Users/xuzhiyuan/CLionProjects/xflow/src/io/load_data_from_disk.cc

src/io/CMakeFiles/io.dir/load_data_from_disk.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/io.dir/load_data_from_disk.cc.i"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/io && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xuzhiyuan/CLionProjects/xflow/src/io/load_data_from_disk.cc > CMakeFiles/io.dir/load_data_from_disk.cc.i

src/io/CMakeFiles/io.dir/load_data_from_disk.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/io.dir/load_data_from_disk.cc.s"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/io && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xuzhiyuan/CLionProjects/xflow/src/io/load_data_from_disk.cc -o CMakeFiles/io.dir/load_data_from_disk.cc.s

# Object files for target io
io_OBJECTS = \
"CMakeFiles/io.dir/load_data_from_disk.cc.o"

# External object files for target io
io_EXTERNAL_OBJECTS =

src/io/libio.a: src/io/CMakeFiles/io.dir/load_data_from_disk.cc.o
src/io/libio.a: src/io/CMakeFiles/io.dir/build.make
src/io/libio.a: src/io/CMakeFiles/io.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/xuzhiyuan/CLionProjects/xflow/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libio.a"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/io && $(CMAKE_COMMAND) -P CMakeFiles/io.dir/cmake_clean_target.cmake
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/io && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/io.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/io/CMakeFiles/io.dir/build: src/io/libio.a

.PHONY : src/io/CMakeFiles/io.dir/build

src/io/CMakeFiles/io.dir/clean:
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/io && $(CMAKE_COMMAND) -P CMakeFiles/io.dir/cmake_clean.cmake
.PHONY : src/io/CMakeFiles/io.dir/clean

src/io/CMakeFiles/io.dir/depend:
	cd /Users/xuzhiyuan/CLionProjects/xflow && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/xuzhiyuan/CLionProjects/xflow /Users/xuzhiyuan/CLionProjects/xflow/src/io /Users/xuzhiyuan/CLionProjects/xflow /Users/xuzhiyuan/CLionProjects/xflow/src/io /Users/xuzhiyuan/CLionProjects/xflow/src/io/CMakeFiles/io.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/io/CMakeFiles/io.dir/depend

