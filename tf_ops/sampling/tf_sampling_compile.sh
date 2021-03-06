#/bin/bash
TF_INC=$(python3.6 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3.6 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
/usr/local/cuda/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I$TF_INC -I /usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ -O2  -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework
