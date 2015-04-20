#!/bin/bash

# dirs
learn_dir=${HOME}/learn
software_dir=${learn_dir}/software
anaconda_dir=${software_dir}/anaconda
htop_dir=${software_dir}/htop
src_dir=${software_dir}/src
htop_src_dir="${src_dir}/htop-1.0.3"

# path
python_src_path="${src_dir}/Anaconda-1.9.2-Linux-x86_64.sh"
htop_tar_path="${src_dir}/htop-1.0.3.tar.gz"

# urls
anaconda_url="http://repo.continuum.io/archive/Anaconda-1.9.2-Linux-x86_64.sh"
htop_url="http://hisham.hm/htop/releases/1.0.3/htop-1.0.3.tar.gz"

# make dirs
mkdir ${learn_dir}
mkdir ${software_dir}
mkdir ${src_dir}
#mkdir ${anaconda_dir}
#mkdir ${htop_dir}

# install Anaconda
#echo "cd ${src_dir}"
#cd ${src_dir}
#
#echo -n "downloading anaconda ... "
#wget ${anaconda_url}
#echo "done"
#
#echo -n "installing anaconda ... "
#bash ${python_src_path} -p ${anaconda_dir} -b
#echo "done"

# install Htop
echo "cd ${src_dir}"
cd ${src_dir}

echo -n "downloading htop ... "
wget ${htop_url}
echo "done"

echo -n "unpacking htop ... "
tar xvf ${htop_tar_path}
echo "done"

echo "cd ${htop_src_dir}"
cd ${htop_src_dir}

echo -n "installing htop ... "
./configure --prefix=${htop_dir}
make
make install
echo "done"
