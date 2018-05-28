# DCC
Automatic Spotting Framework for Detonator Coded Characters based on Convolutional Neural Networks

Code written by Guandong Cen(cenguandong@qq.com)

Installation

1. For caffe version of this project, please install HED(https://github.com/s9xie/hed) at first. 

2. Add the following code in 'caffe.proto'.
  optional PostParameter post_param = 139;
  optional JaccardLossParameter jaccard_loss_param = 141;
  
  message JaccardLossParameter {
    optional float w_ = 1 [default = 1.0];
  }
  message PostParameter {
    optional float binary_threshold = 1 [default = 0.7];
    optional float area_threshold = 2 [default = 0.015625];
    optional float mean_h = 3 [default = 35.0];
    optional float mean_w = 4 [default = 258.0];
    enum Lt {
      SIGMOID = 5;
      JACCARD = 6;
    }
    optional Lt losstype = 7 [default = JACCARD];
  }

3. Replace file 'vision_layers.hpp' and file 'loss_layers.hpp' in '$CAFFE_ROOT/include/caffe/'.

4. Add these layers: BatchNorm Layer(not provided in this branch), jaccard_loss_layer.cpp and post_layer.cpp.

5. make all & run deploy_demo/demo_e2e.py.
