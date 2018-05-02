# DCC
Automatic Spotting Framework for Detonator Coded Characters based on Convolutional Neural Networks

Code written by Guandong Cen(cenguandong@qq.com)

Installation

1.For caffe version of this project, please install HED(https://github.com/s9xie/hed) at first. 

2.Replace file 'vision_layers.hpp' and file 'loss_layers.hpp'.

3.Add these layers: BatchNorm Layer(not provided in this branch), jaccard_loss_layer.cpp and post_layer.cpp.

4.make all & run deploy_demo/demo_e2e.py.
