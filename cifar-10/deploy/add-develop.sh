#!/bin/bash

# local dirs
local_src_dir=${HOME}/learn-deploy/src-data

# remote dirs
root_dir=/home/admin/learn
software_dir=${root_dir}/software
src_dir=${root_dir}/src
instance_dir=${src_dir}/${3}
script_dir=${instance_dir}/script
csv_dir=${instance_dir}/csv
log_dir=${instance_dir}/log

# paths
python_path=${software_dir}/anaconda/bin/python
csv_path=${csv_dir}/develop.csv
log_path=${log_dir}/develop.log
script_path=${script_dir}/develop.py
local_config_path=${1}
target_config_path=${instance_dir}/config/develop.cfg

# prepare for src and data
ssh admin@${2} "mkdir -p ${src_dir}"
echo -n "copy src and data ... "
scp -r ${local_src_dir} ${2}:${instance_dir}
echo "done"

# prepare for config
echo -n "copy develop.cfg ... "
scp ${local_config_path} ${2}:${target_config_path}
echo "done"

# start
ssh ${2} "cd ${script_dir}; nohup ${python_path} ${script_path} > ${csv_path} 2>&1 &"

