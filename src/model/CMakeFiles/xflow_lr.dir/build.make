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
include src/model/CMakeFiles/xflow_lr.dir/depend.make

# Include the progress variables for this target.
include src/model/CMakeFiles/xflow_lr.dir/progress.make

# Include the compile flags for this target's objects.
include src/model/CMakeFiles/xflow_lr.dir/flags.make

src/model/CMakeFiles/xflow_lr.dir/main.cc.o: src/model/CMakeFiles/xflow_lr.dir/flags.make
src/model/CMakeFiles/xflow_lr.dir/main.cc.o: src/model/main.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xuzhiyuan/CLionProjects/xflow/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/model/CMakeFiles/xflow_lr.dir/main.cc.o"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/model && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/xflow_lr.dir/main.cc.o -c /Users/xuzhiyuan/CLionProjects/xflow/src/model/main.cc

src/model/CMakeFiles/xflow_lr.dir/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/xflow_lr.dir/main.cc.i"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/model && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xuzhiyuan/CLionProjects/xflow/src/model/main.cc > CMakeFiles/xflow_lr.dir/main.cc.i

src/model/CMakeFiles/xflow_lr.dir/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/xflow_lr.dir/main.cc.s"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/model && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xuzhiyuan/CLionProjects/xflow/src/model/main.cc -o CMakeFiles/xflow_lr.dir/main.cc.s

src/model/CMakeFiles/xflow_lr.dir/lr/lr_worker.cc.o: src/model/CMakeFiles/xflow_lr.dir/flags.make
src/model/CMakeFiles/xflow_lr.dir/lr/lr_worker.cc.o: src/model/lr/lr_worker.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xuzhiyuan/CLionProjects/xflow/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/model/CMakeFiles/xflow_lr.dir/lr/lr_worker.cc.o"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/model && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/xflow_lr.dir/lr/lr_worker.cc.o -c /Users/xuzhiyuan/CLionProjects/xflow/src/model/lr/lr_worker.cc

src/model/CMakeFiles/xflow_lr.dir/lr/lr_worker.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/xflow_lr.dir/lr/lr_worker.cc.i"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/model && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xuzhiyuan/CLionProjects/xflow/src/model/lr/lr_worker.cc > CMakeFiles/xflow_lr.dir/lr/lr_worker.cc.i

src/model/CMakeFiles/xflow_lr.dir/lr/lr_worker.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/xflow_lr.dir/lr/lr_worker.cc.s"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/model && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xuzhiyuan/CLionProjects/xflow/src/model/lr/lr_worker.cc -o CMakeFiles/xflow_lr.dir/lr/lr_worker.cc.s

src/model/CMakeFiles/xflow_lr.dir/fm/fm_worker.cc.o: src/model/CMakeFiles/xflow_lr.dir/flags.make
src/model/CMakeFiles/xflow_lr.dir/fm/fm_worker.cc.o: src/model/fm/fm_worker.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xuzhiyuan/CLionProjects/xflow/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/model/CMakeFiles/xflow_lr.dir/fm/fm_worker.cc.o"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/model && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/xflow_lr.dir/fm/fm_worker.cc.o -c /Users/xuzhiyuan/CLionProjects/xflow/src/model/fm/fm_worker.cc

src/model/CMakeFiles/xflow_lr.dir/fm/fm_worker.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/xflow_lr.dir/fm/fm_worker.cc.i"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/model && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xuzhiyuan/CLionProjects/xflow/src/model/fm/fm_worker.cc > CMakeFiles/xflow_lr.dir/fm/fm_worker.cc.i

src/model/CMakeFiles/xflow_lr.dir/fm/fm_worker.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/xflow_lr.dir/fm/fm_worker.cc.s"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/model && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xuzhiyuan/CLionProjects/xflow/src/model/fm/fm_worker.cc -o CMakeFiles/xflow_lr.dir/fm/fm_worker.cc.s

src/model/CMakeFiles/xflow_lr.dir/mvm/mvm_worker.cc.o: src/model/CMakeFiles/xflow_lr.dir/flags.make
src/model/CMakeFiles/xflow_lr.dir/mvm/mvm_worker.cc.o: src/model/mvm/mvm_worker.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/xuzhiyuan/CLionProjects/xflow/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/model/CMakeFiles/xflow_lr.dir/mvm/mvm_worker.cc.o"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/model && /Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/xflow_lr.dir/mvm/mvm_worker.cc.o -c /Users/xuzhiyuan/CLionProjects/xflow/src/model/mvm/mvm_worker.cc

src/model/CMakeFiles/xflow_lr.dir/mvm/mvm_worker.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/xflow_lr.dir/mvm/mvm_worker.cc.i"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/model && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/xuzhiyuan/CLionProjects/xflow/src/model/mvm/mvm_worker.cc > CMakeFiles/xflow_lr.dir/mvm/mvm_worker.cc.i

src/model/CMakeFiles/xflow_lr.dir/mvm/mvm_worker.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/xflow_lr.dir/mvm/mvm_worker.cc.s"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/model && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/xuzhiyuan/CLionProjects/xflow/src/model/mvm/mvm_worker.cc -o CMakeFiles/xflow_lr.dir/mvm/mvm_worker.cc.s

# Object files for target xflow_lr
xflow_lr_OBJECTS = \
"CMakeFiles/xflow_lr.dir/main.cc.o" \
"CMakeFiles/xflow_lr.dir/lr/lr_worker.cc.o" \
"CMakeFiles/xflow_lr.dir/fm/fm_worker.cc.o" \
"CMakeFiles/xflow_lr.dir/mvm/mvm_worker.cc.o"

# External object files for target xflow_lr
xflow_lr_EXTERNAL_OBJECTS =

test/src/xflow_lr: src/model/CMakeFiles/xflow_lr.dir/main.cc.o
test/src/xflow_lr: src/model/CMakeFiles/xflow_lr.dir/lr/lr_worker.cc.o
test/src/xflow_lr: src/model/CMakeFiles/xflow_lr.dir/fm/fm_worker.cc.o
test/src/xflow_lr: src/model/CMakeFiles/xflow_lr.dir/mvm/mvm_worker.cc.o
test/src/xflow_lr: src/model/CMakeFiles/xflow_lr.dir/build.make
test/src/xflow_lr: src/io/libio.a
test/src/xflow_lr: ps-lite/libpslite.a
test/src/xflow_lr: src/model/CMakeFiles/xflow_lr.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/xuzhiyuan/CLionProjects/xflow/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable ../../test/src/xflow_lr"
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/model && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/xflow_lr.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/model/CMakeFiles/xflow_lr.dir/build: test/src/xflow_lr

.PHONY : src/model/CMakeFiles/xflow_lr.dir/build

src/model/CMakeFiles/xflow_lr.dir/clean:
	cd /Users/xuzhiyuan/CLionProjects/xflow/src/model && $(CMAKE_COMMAND) -P CMakeFiles/xflow_lr.dir/cmake_clean.cmake
.PHONY : src/model/CMakeFiles/xflow_lr.dir/clean

src/model/CMakeFiles/xflow_lr.dir/depend:
	cd /Users/xuzhiyuan/CLionProjects/xflow && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/xuzhiyuan/CLionProjects/xflow /Users/xuzhiyuan/CLionProjects/xflow/src/model /Users/xuzhiyuan/CLionProjects/xflow /Users/xuzhiyuan/CLionProjects/xflow/src/model /Users/xuzhiyuan/CLionProjects/xflow/src/model/CMakeFiles/xflow_lr.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/model/CMakeFiles/xflow_lr.dir/depend

