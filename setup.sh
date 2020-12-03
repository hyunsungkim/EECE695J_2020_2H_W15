#!/bin/bash
# Install tensorflow 2 dependencies
sudo apt-get update
sudo apt-get -y dist-upgrade
sudo apt-get install -y libopenblas-dev libblas-dev m4 cmake cython python3-dev python3-yaml python3-setuptools git

pip install --upgrade pip
pip3 install --upgrade --use-feature=2020-resolver setuptools
pip3 install numpy==1.19.0

sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev gcc gfortran python-dev libgfortran5 libatlas3-base libatlas-base-dev libopenblas-dev libopenblas-base libblas-dev liblapack-dev cython libatlas-base-dev openmpi-bin libopenmpi-dev python3-dev vim

# Install tensorflow-related python libraries
pip3 install keras_applications==1.0.8 --no-deps
pip3 install keras_preprocessing==1.1.0 --no-deps
pip3 install h5py==2.9.0
pip3 install pybind11
pip3 install -U --user  --use-feature=2020-resolver six wheel mock

# Install tensorflow 2 from pre-built wheel
cd
wget "https://raw.githubusercontent.com/PINTO0309/Tensorflow-bin/master/tensorflow-2.3.0-cp37-none-linux_armv7l_download.sh"
sudo chmod +x tensorflow-2.3.0-cp37-none-linux_armv7l_download.sh
./tensorflow-2.3.0-cp37-none-linux_armv7l_download.sh
pip3 uninstall -y tensorflow
pip3 install tensorflow-2.3.0-cp37-none-linux_armv7l.whl
pip3 install matplotlib

# Install jupyterlab
pip3 install setuptools
sudo apt -y install libffi-dev
pip3 install cffi
pip3 install jupyterlab
cd

## For Tensorflow2 installation, refer to https://itnext.io/installing-tensorflow-2-3-0-for-raspberry-pi3-4-debian-buster-11447cb31fc4
## For Jupyterlab installation, refer to https://medium.com/analytics-vidhya/jupyter-lab-on-raspberry-pi-22876591b227
