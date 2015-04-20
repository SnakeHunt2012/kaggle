#!/bin/bash

# dirs
deploy_dir=${HOME}/learn-deploy
stdout_dir=${deploy_dir}/stdout.d
stderr_dir=${deploy_dir}/stdrrr.d
software_dir="/home/admin/learn/software"
local_anaconda_dir=${deploy_dir}/software/anaconda
target_anaconda_dir="/home/admin/learn/software/anaconda"

# paths
iplist_path=${deploy_dir}/ip.list
local_setenv_path=${deploy_dir}/set-env.sh
target_env_path="/home/admin/set-env.sh"
local_anaconda_path=${deploy_dir}/software/anaconda.tar.gz
target_anaconda_path="/home/admin/learn/software/src/anaconda.tar.gz"

# setting enviroment
pssh -h ${iplist_path} -v -P -t 0 -o ${stdout_dir} -e ${stderr_dir} "rm -rf ${target_env_path}"
pscp -h ${iplist_path} ${local_setenv_path}  ${target_env_path}
pssh -h ${iplist_path} -v -P -t 0 -o ${stdout_dir} -e ${stderr_dir} "chmod +x ${target_env_path}"
pssh -h ${iplist_path} -v -P -t 0 -o ${stdout_dir} -e ${stderr_dir} "${target_env_path}"

# copy anaconda
pscp -r -h ${iplist_path} ${local_anaconda_path} ${target_anaconda_path}
pssh -h ${iplist_path} -v -t 0 -o ${stdout_dir} -e ${stderr_dir} "tar xvf ${target_anaconda_path} -C ${software_dir}/"

