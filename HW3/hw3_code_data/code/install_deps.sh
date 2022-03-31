# Necessary: pip install -r requirements will build then install, so if numpy is not existing the procedure will fail.
pip install numpy
pip install cffi
pip install scipy
pip install matplotlib
pip install wheel
pip install tqdm

pip install git+https://github.com/theNded/PySPQR.git

# # CMake
# sudo apt-get install cmake -y
# # google-glog + gflags
# sudo apt-get install libgoogle-glog-dev libgflags-dev -y 
# # BLAS & LAPACK
# sudo apt-get install libatlas-base-dev -y
# # Eigen3
# sudo apt-get install libeigen3-dev -y 
# # SuiteSparse and CXSparse (optional)
# sudo apt-get install libsuitesparse-dev -y