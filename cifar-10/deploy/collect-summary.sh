#!/bin/bash

# dirs
root_dir=$(pwd)

# paths
iplist_path=${root_dir}/ip.list

# strings
command="ps -ef | grep \"develop.py\" |\
         grep -v \"grep develop.py\" |\
         awk '{print \$9}' |\
         awk -F '/' '{print \$6}'"

# collect summary
for ip in $(cat ${iplist_path})
do
    #echo 
	#echo "------------------------- IP: ${ip} -------------------------"
	info=$(ssh -q ${ip} ${command})
	for meta in ${info}
	do
		learning_rate=$(echo ${meta} | awk -F '_' '{print $1}')
		batch_size=$(echo ${meta} | awk -F '_' '{print $2}')
		n_hidden=$(echo ${meta} | awk -F '_' '{print $3}')
		L2=$(echo ${meta} | awk -F '_' '{print $4}')
		activation=$(echo ${meta} | awk -F '_' '{print $5}')
		echo -e "\t${learning_rate}\t${batch_size}\t${n_hidden}\t${L2}\t${activation}\t${ip}"
	done
done
