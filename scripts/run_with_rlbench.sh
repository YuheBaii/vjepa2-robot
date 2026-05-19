#!/bin/bash
# Wrapper script that sets CoppeliaSim environment before running Python.
export COPPELIASIM_ROOT=/home/yuhe/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH=$COPPELIASIM_ROOT:$LD_LIBRARY_PATH
export QT_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=${DISPLAY:-:1}

exec /home/yuhe/miniforge3/envs/vjepa2-robot/bin/python -u "$@"
