#!/usr/bin/python
from __future__ import absolute_import
from __future__ import print_function
from six.moves import range

# TODO: not needing all of these imports. cleanup
import argparse
import os, os.path
import re
import errno
import sys
import time
import subprocess
from random import randint
from io import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image 
import cv2
from deep_dream import DeepDream

LIST_IDS = [21, 23, 24, 25, 27, 28, 30, 31]
START_DREAM = 4  # s

# GRB: why not just define the neural network here instead?
net = None # will become global reference to the network model once inside the loop

# viewport
viewport_w, viewport_h = 1900, 1080  # 1280,720 # display resolution
b_debug = False
font = cv2.FONT_HERSHEY_PLAIN
white = (255,255,255)
b_showMotionDetect = False # flag for motion detection view

# camera object
cap = cv2.VideoCapture(0)
cap_w, cap_h = 1280,720 # capture resolution
cap.set(3,cap_w)
cap.set(4,cap_h)

# -------
# utility
# ------- 

class MotionDetector(object):
    # cap: global capture object
    def __init__(self, delta_count_threshold=50000):
        self.delta_count_threshold = delta_count_threshold
        self.delta_count = 0
        self.delta_count_last = 0
        self.reset()
#        self.t_minus = cap.read()[1] 
#        self.t_now = cap.read()[1]
#        self.t_plus = cap.read()[1]
        self.width = cap.get(3)
        self.height = cap.get(4)
        self.delta_view = np.zeros((int(cap.get(4)), int(cap.get(3)) ,3), np.uint8) # empty img
        self.isMotionDetected = False
#        self.isMotionDetected_last = False
    def delta_images(self,t0, t1):
        d1 = cv2.absdiff(t1, t0)
        return d1
    
    def reset(self):
        self.t_init = self.capture()
    
    def capture(self):
        return cv2.flip(cap.read()[1], 1)
    
    def process(self):
        #print 'processing'
        
        self.t_now = self.capture()
        self.delta_view = self.delta_images(self.t_init, self.t_now)
        retval, self.delta_view = cv2.threshold(self.delta_view, 50, 255, 3)
        cv2.normalize(self.delta_view, self.delta_view, 0, 255, cv2.NORM_MINMAX)
        img_count_view = cv2.cvtColor(self.delta_view, cv2.COLOR_RGB2GRAY)
        self.delta_count = cv2.countNonZero(img_count_view)
#        self.delta_view = cv2.flip(self.delta_view, 1)
        if (self.delta_count >= self.delta_count_threshold):
            print("+ CHANGE DETECTED ({0})".format(self.delta_count))
            self.isMotionDetected = True
        else:
            print("+ no change ({0})".format(self.delta_count))
            self.isMotionDetected = False
#        print('Movement:', self.delta_count > self.delta_count_threshold)
    def isResting(self):
        return not self.isMotionDetected


# this creates a RGB image from our image matrix
# GRB: what is the expected input here??
# GRB: why do I have 2 functions (below) to write images to the display?
def showarray(window_name, a):
    global b_debug, DELTA_COUNT_THRESHOLD, b_showMotionDetect

    # convert and clip our floating point matrix into 0-255 values for RGB image
    a = np.uint8(np.clip(a, 0, 255))

    # resize takes its arguments as w,h in that order
    dim = (viewport_w, viewport_h)
    a = cv2.resize(a, dim, interpolation = cv2.INTER_LINEAR)

    # write to window
    cv2.imshow(window_name, a)
    # weighted addition the input to buffer2
    #   the usual ratio between the input at buffer2 is 1:0
    #   if we knew when Tracker.isResting() had just toggled to false
    #       we would start a timer
    #           we could increment the ratio of input to buffer2 from 1:1 to 1:0
    #           if the Tracker.isResting() state chaged to False again while we were doing this
    #               we would write the result of our weighted addition to buffer2
    #               we would re-set the ratio to be 1:1
    # is it expensive to perform a weighted addition between 2 images each frame

    # refresh the display 
    key = cv2.waitKey(1) & 0xFF

    if key == 27: # Escape key: Exit
        sys.exit()
    elif key == 96: # `(tilde) key: toggle HUD
        b_debug = not b_debug
    elif key == 43: # + key : increase motion threshold
        Tracker.delta_count_threshold += 1000
        print((Tracker.delta_count_threshold))
    elif key == 45: # - key : decrease motion threshold
        Tracker.delta_count_threshold -= 1000
        if Tracker.delta_count_threshold < 1:
            Tracker.delta_count_threshold = 0
        print((Tracker.delta_count_threshold))
    elif key == 49: # 1 key : toggle motion detect window
        b_showMotionDetect = not b_showMotionDetect
        if b_showMotionDetect:
            cv2.namedWindow('deltaview',cv2.WINDOW_AUTOSIZE)
        else:
            cv2.destroyWindow('delta_view')


# -------
# implements forward and backward passes thru the network
# apply normalized ascent step upon the image in the networks data blob
# ------- 
def make_step(net, step_size=1.5, end='inception_4c/output',jitter=32, clip=True):
    src = net.blobs['data']     # input image is stored in Net's 'data' blob
    dst = net.blobs[end]        # destination is the end layer specified by argument

    ox, oy = np.random.randint(-jitter, jitter + 1, 2)          # calculate jitter
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    # this bit is where the neural net runs the computation
    net.forward(end=end)    # make sure we stop on the chosen neural layer
    dst.diff[:] = dst.data  # specify the optimization objective
    net.backward(start=end) # backwards propagation
    g = src.diff[0]         # store the error 

    # apply normalized ascent step to the input image and get closer to our target state
    src.data[:] += step_size / np.abs(g).mean() * g

    # unshift image jitter              
    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2)   

    # subtract image mean and clip our matrix to the values
    bias = net.transformer.mean['data']
    src.data[:] = np.clip(src.data, -bias, 255-bias)

# -------
# sets up image buffers and octave structure for iterating thru and amplifying neural output
# iterates ththru the neural network 
# REM sleep, in other words
# ------- 
def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True, **step_params):
    
    '''
    counter = 0
    while counter < 50 and Tracker.isResting():
        Tracker.process()
        print counter
        counter += 1
    # if Tracker.isResting == False
    # copy current webcam frame into buffer2 (this would be the net blob in real life)
    #       where is that stored?
    #           should it be stored in this "buffer2" by default?
    #       where was it captured?

    return cap.read()[1]
    '''
    
    Tracker.reset()

    # before doing anything check the current value of Tracker.isResting()
    # we sampled the webcam right before calling this function
    if not Tracker.isResting():
        time.sleep(0.1)
        return Tracker.t_init
    
    
    from PIL import Image
#    img = Image.fromarray(base_img)
    img = Image.fromarray(np.uint8(Tracker.t_init*255))
    out = np.array(img)
        
    i = 0
    
    
    layer_id = LIST_IDS[np.random.randint(len(LIST_IDS))]  # change the kind of dream
    Dreamer.change_id(layer_id=layer_id)
    print('Dreamer Layer {0}'.format(layer_id))
    while Tracker.isResting():
        
        if i > iter_n:
            Tracker.process()
            continue    # do nothing after convergence (but stay on same image)

        # calls the neural net step function
#        try:
        img = Dreamer.load(img).deepDreamProcess()
#        except RuntimeError:
#           from torch.cuda import empty_cache
#           empty_cache() 
#           img = DeepDream(img).deepDreamProcess()
        out = np.array(img)

#        
#        #         zoom progressively on the frame
#        z = 0.002
#        out = out*(255.0/np.percentile(cap.read()[1], 98))
#        out = nd.affine_transform(out, [1-i*z,1-i*z,1], [cap_h*i*z/2,cap_w*i*z/2,0], order=1)

        showarray('Dream',out) #src.data[0])
        Tracker.process()

        # increment
        i += 1
        
    # if movement, return last one
    return out
        

# -------
# MAIN
# ------- 
def main(iterations, stepsize, octaves, octave_scale, end):
    global net

    # start timer
    print('+ TIMER START :REM.main')
    now = time.time()

    # set GPU mode
#    caffe.set_device(0)       # HACK
#    caffe.set_mode_gpu()      # HACK

    cv2.namedWindow('Dream',cv2.WINDOW_AUTOSIZE)

#    # parameters
#    model_path = 'E:/Users/Gary/Documents/code/models/cars/'
#    net_fn = model_path + 'deploy.prototxt'
#    param_fn = model_path + 'googlenet_finetune_web_car_iter_10000.caffemodel'

    nrframes = 1
    jitter = int(cap_w/2)
    zoom = 1

    if iterations is None: iterations = 30
    if stepsize is None: stepsize = 1.5
    if octaves is None: octaves = 4
    if octave_scale is None: octave_scale = 1.8
    if end is None: end = 'inception_5a_3x3'

    print('[main] iterations:{arg1} step size:{arg2} octaves:{arg3} octave_scale:{arg4} end:{arg5}'.format(arg1=iterations,arg2=stepsize,arg3=octaves,arg4=octave_scale,arg5=end))

    frame = Tracker.capture()  # initial camera image for init
    showarray('Dream',frame)
#    s = 0.001 # scale coefficient for uninterrupted dreaming
    still_time = 0
    t0 = time.time()
    while True:
##         zoom in a bit on the frame
#        frame = frame*(255.0/np.percentile(cap.read()[1], 98))
#        frame = nd.affine_transform(frame, [1-s,1-s,1], [cap_h*s/2,cap_w*s/2,0], order=1)

        Tracker.process()

        
        if Tracker.isResting():
            if still_time < START_DREAM:
                showarray('Dream',Tracker.t_now)
                still_time += time.time() - t0
            else:
                print('still_time', still_time)
                # kicks off rem sleep - will begin continual iteration of the image through the model
                frame = deepdream(net, frame, iter_n = iterations, octave_n = octaves, octave_scale = octave_scale, step_size = stepsize, end = end)
                still_time = 0
        else:  # show difference
            showarray('Dream',Tracker.delta_view)
            frame = frame   # does not change
            Tracker.reset()
            still_time = 0
        t0 = time.time()

# -------- 
# INIT
# --------
Tracker = MotionDetector() # motion detector object
Dreamer = DeepDream()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='REM')
    parser.add_argument(
        '-e','--end',
        required=False,
        help='End layer. Default: inception_4c/output')
    parser.add_argument(
        '-oct','--octaves',
        type=int,
        required=False,
        help='Octaves. Default: 4')
    parser.add_argument(
        '-octs','--octavescale',
        type=float,
        required=False,
        help='Octave Scale. Default: 1.4',)
    parser.add_argument(
        '-i','--iterations',
        type=int,
        required=False,
        help='Iterations. Default: 10')
    parser.add_argument(
        '-s','--stepsize',
        type=float,
        required=False,
        help='Step Size. Default: 1.5')
    parser.add_argument(
        '-c','--clip',
        type=float,
        required=False,
        help='Step Size. Default: 1.5')
    args = parser.parse_args()
    main(args.iterations, args.stepsize, args.octaves,  args.octavescale, args.end)
