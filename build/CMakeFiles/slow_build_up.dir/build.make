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
include CMakeFiles/slow_build_up.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/slow_build_up.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/slow_build_up.dir/flags.make

CMakeFiles/slow_build_up.dir/slow_build_up.cpp.o: CMakeFiles/slow_build_up.dir/flags.make
CMakeFiles/slow_build_up.dir/slow_build_up.cpp.o: ../slow_build_up.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/valentin/Master_Thesis/Coding/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/slow_build_up.dir/slow_build_up.cpp.o"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/slow_build_up.dir/slow_build_up.cpp.o -c /home/valentin/Master_Thesis/Coding/slow_build_up.cpp

CMakeFiles/slow_build_up.dir/slow_build_up.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/slow_build_up.dir/slow_build_up.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/valentin/Master_Thesis/Coding/slow_build_up.cpp > CMakeFiles/slow_build_up.dir/slow_build_up.cpp.i

CMakeFiles/slow_build_up.dir/slow_build_up.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/slow_build_up.dir/slow_build_up.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/valentin/Master_Thesis/Coding/slow_build_up.cpp -o CMakeFiles/slow_build_up.dir/slow_build_up.cpp.s

# Object files for target slow_build_up
slow_build_up_OBJECTS = \
"CMakeFiles/slow_build_up.dir/slow_build_up.cpp.o"

# External object files for target slow_build_up
slow_build_up_EXTERNAL_OBJECTS =

slow_build_up: CMakeFiles/slow_build_up.dir/slow_build_up.cpp.o
slow_build_up: CMakeFiles/slow_build_up.dir/build.make
slow_build_up: ../libtorch/lib/libtorch.so
slow_build_up: ../libtorch/lib/libc10.so
slow_build_up: ../libtorch/lib/libkineto.a
slow_build_up: ../libtorch/lib/libc10.so
slow_build_up: CMakeFiles/slow_build_up.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/valentin/Master_Thesis/Coding/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable slow_build_up"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/slow_build_up.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/slow_build_up.dir/build: slow_build_up

.PHONY : CMakeFiles/slow_build_up.dir/build

CMakeFiles/slow_build_up.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/slow_build_up.dir/cmake_clean.cmake
.PHONY : CMakeFiles/slow_build_up.dir/clean

CMakeFiles/slow_build_up.dir/depend:
	cd /home/valentin/Master_Thesis/Coding/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/valentin/Master_Thesis/Coding /home/valentin/Master_Thesis/Coding /home/valentin/Master_Thesis/Coding/build /home/valentin/Master_Thesis/Coding/build /home/valentin/Master_Thesis/Coding/build/CMakeFiles/slow_build_up.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/slow_build_up.dir/depend

