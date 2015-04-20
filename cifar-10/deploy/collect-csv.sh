#!/bin/bash

# dirs
work_dir=$(pwd)
csv_dir=${work_dir}/csv
root_dir=/home/admin/learn
src_dir=${root_dir}/src

# paths
iplist_path=${work_dir}/ip.list	

# collect dsv
rm -rf ${csv_dir}
mkdir ${csv_dir}
for ip in $(cat ${iplist_path})
do
	echo "come into ${ip}:"
	for config in $(ssh ${ip} "ls ${src_dir}")
	do
		echo -n "copy ${config} ... "
		scp ${ip}:${src_dir}/${config}/csv/develop.csv ${csv_dir}/${config}.csv
		echo "done"
	done
done

