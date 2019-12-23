TF_INC=$(python3.6 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3.6 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I $TF_INC -I /usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -I $TF_INC/external/nsync/public -L $TF_LIB -ltensorflow_framework

