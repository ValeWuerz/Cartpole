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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/valentin/Master_Thesis/Coding

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/valentin/Master_Thesis/Coding/build

# Include any dependencies generated for this target.
include CMakeFiles/deepl_actions.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/deepl_actions.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/deepl_actions.dir/flags.make

CMakeFiles/deepl_actions.dir/deepl_actions.cpp.o: CMakeFiles/deepl_actions.dir/flags.make
CMakeFiles/deepl_actions.dir/deepl_actions.cpp.o: ../deepl_actions.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/valentin/Master_Thesis/Coding/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/deepl_actions.dir/deepl_actions.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/deepl_actions.dir/deepl_actions.cpp.o -c /home/valentin/Master_Thesis/Coding/deepl_actions.cpp

CMakeFiles/deepl_actions.dir/deepl_actions.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/deepl_actions.dir/deepl_actions.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/valentin/Master_Thesis/Coding/deepl_actions.cpp > CMakeFiles/deepl_actions.dir/deepl_actions.cpp.i

CMakeFiles/deepl_actions.dir/deepl_actions.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/deepl_actions.dir/deepl_actions.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/valentin/Master_Thesis/Coding/deepl_actions.cpp -o CMakeFiles/deepl_actions.dir/deepl_actions.cpp.s

# Object files for target deepl_actions
deepl_actions_OBJECTS = \
"CMakeFiles/deepl_actions.dir/deepl_actions.cpp.o"

# External object files for target deepl_actions
deepl_actions_EXTERNAL_OBJECTS =

deepl_actions: CMakeFiles/deepl_actions.dir/deepl_actions.cpp.o
deepl_actions: CMakeFiles/deepl_actions.dir/build.make
deepl_actions: ../libtorch/lib/libtorch.so
deepl_actions: ../libtorch/lib/libc10.so
deepl_actions: ../libtorch/lib/libkineto.a
deepl_actions: ../libtorch/lib/libc10.so
deepl_actions: CMakeFiles/deepl_actions.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/valentin/Master_Thesis/Coding/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable deepl_actions"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/deepl_actions.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/deepl_actions.dir/build: deepl_actions

.PHONY : CMakeFiles/deepl_actions.dir/build

CMakeFiles/deepl_actions.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/deepl_actions.dir/cmake_clean.cmake
.PHONY : CMakeFiles/deepl_actions.dir/clean

CMakeFiles/deepl_actions.dir/depend:
	cd /home/valentin/Master_Thesis/Coding/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/valentin/Master_Thesis/Coding /home/valentin/Master_Thesis/Coding /home/valentin/Master_Thesis/Coding/build /home/valentin/Master_Thesis/Coding/build /home/valentin/Master_Thesis/Coding/build/CMakeFiles/deepl_actions.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/deepl_actions.dir/depend

