# -*- coding: utf-8 -*-
import sys
sys.path.append('/home/cen/caffe/python')
import caffe
import cv2
import numpy as np
import os
import datetime
import shutil

net = caffe.Net('deploy_jaccard.prototxt', 'deploy_e2e.caffemodel', caffe.TEST) 
caffe.set_mode_cpu()

currentCount = 0
error = 0
allTime = 0.0

testDir = 'img/'

imageList = os.listdir(testDir)
for i in imageList[:]:
    if os.path.splitext(i)[1] == '.jpg':
        begin = datetime.datetime.now()
        currentCount += 1
        input_image = caffe.io.load_image(testDir + i)
        hs,ws,ch = input_image.shape
        net.blobs['data'].reshape(1,ch,hs,ws)
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2,0,1))
        transformer.set_channel_swap('data', (2,1,0))
        transformer.set_raw_scale('data', 255)
        net.blobs['data'].data[...]  = transformer.preprocess('data', input_image)   
        output = net.forward()
        outp = [1]*13

        outp[0] = str(output['prob_01'][0:10].argmax())
        outp[1] = str(output['prob_02'][0:10].argmax())
        outp[2] = str(output['prob_03'][0:10].argmax())
        outp[3] = str(output['prob_04'][0:10].argmax())
        outp[4] = str(output['prob_05'][0:10].argmax())
        outp[5] = str(output['prob_06'][0:10].argmax())
        outp[6] = str(output['prob_07'][0:10].argmax())
        outp[7] = str(output['prob_08'][0].argmax())
        outp[8] = str(output['prob_09'][0:10].argmax())
        outp[9] = str(output['prob_10'][0:10].argmax())
        outp[10] = str(output['prob_11'][0:10].argmax())
        outp[11] = str(output['prob_12'][0:10].argmax())
        outp[12] = str(output['prob_13'][0:10].argmax())        

        for k in range(13):
            o = outp[k]
            if(int(o) == 10):
                outp[k] = 'A'
            elif(int(o) == 11):
                outp[k] = 'H'
            elif(int(o) == 12):
                outp[k] = 'X'
            elif(int(o) == 13):
                outp[k] = 'S'
       
        predictStr = str(outp[0]+outp[1]+outp[2]+outp[3]+outp[4]+outp[5]+outp[6]+outp[7]+outp[8]+outp[9]+outp[10]+outp[11]+outp[12])


        if(predictStr != (i.split('.')[0])):
            error += 1;
            print i,predictStr
        else:
            pass
 
            
        end = datetime.datetime.now()
        print str(end-begin)[6:]
        allTime += float(str(end-begin)[6:])
        print currentCount,error,1 - (float(error)/float(currentCount))


