#!/bin/bash

# dirs
deploy_dir=${HOME}/learn-deploy
learn_dir=/home/admin/learn

# paths
iplist_path=${deploy_dir}/ip.list

# clean env
pssh -h ${iplist_path} -v -P -t 0 -o ${stdout_dir} -e ${stderr_dir} "rm -rf ${learn_dir}"
