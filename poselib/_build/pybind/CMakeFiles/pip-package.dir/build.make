# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

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
CMAKE_COMMAND = /home/guan/anaconda3/lib/python3.7/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/guan/anaconda3/lib/python3.7/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib/_build

# Utility rule file for pip-package.

# Include any custom commands dependencies for this target.
include pybind/CMakeFiles/pip-package.dir/compiler_depend.make

# Include the progress variables for this target.
include pybind/CMakeFiles/pip-package.dir/progress.make

pybind/CMakeFiles/pip-package: PoseLib/libPoseLib.a
	cd /media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib/_build/pybind && /home/guan/anaconda3/lib/python3.7/site-packages/cmake/data/bin/cmake -DPYTHON_PACKAGE=ON -DBUILD_SHARED_LIBS=OFF /media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib/_build
	cd /media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib/_build/pybind && /home/guan/anaconda3/bin/python3.7 /media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib/_build/pybind/setup.py bdist_wheel --dist-dir pip_package
	cd /media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib/_build/pybind && echo pip\ wheel\ created\ at\ /media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib/_build/pybind/pip_package

pip-package: pybind/CMakeFiles/pip-package
pip-package: pybind/CMakeFiles/pip-package.dir/build.make
.PHONY : pip-package

# Rule to build all files generated by this target.
pybind/CMakeFiles/pip-package.dir/build: pip-package
.PHONY : pybind/CMakeFiles/pip-package.dir/build

pybind/CMakeFiles/pip-package.dir/clean:
	cd /media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib/_build/pybind && $(CMAKE_COMMAND) -P CMakeFiles/pip-package.dir/cmake_clean.cmake
.PHONY : pybind/CMakeFiles/pip-package.dir/clean

pybind/CMakeFiles/pip-package.dir/depend:
	cd /media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib/_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib /media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib/pybind /media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib/_build /media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib/_build/pybind /media/guan/3CD61590D6154C10/SomeCodes/sensloc_new/poselib/_build/pybind/CMakeFiles/pip-package.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : pybind/CMakeFiles/pip-package.dir/depend
