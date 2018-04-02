#! /usr/bin/env python3

# Copyright(c) 2017 Intel Corporation.
# License: MIT See LICENSE file in root directory.


from mvnc import mvncapi as mvnc
import sys
import numpy
import cv2
import time
import csv
import os
import sys

from sys import argv

# name of the opencv window
cv_window_name = "vehicle platooning"

# labels AKA classes.  The class IDs returned
# are the indices into this list
labels = ('background',
          'aeroplane', 'bicycle', 'bird', 'boat',
          'bottle', 'bus', 'car', 'cat', 'chair',
          'cow', 'diningtable', 'dog', 'horse',
          'motorbike', 'person', 'pottedplant',
          'sheep', 'sofa', 'train', 'tvmonitor')

# the ssd mobilenet image width and height
NETWORK_IMAGE_WIDTH = 300
NETWORK_IMAGE_HEIGHT = 300

# the minimal score for a box to be shown
min_score_percent = 60

# the resize_window arg will modify these if its specified on the commandline
resize_output = False
resize_output_width = 0
resize_output_height = 0

# read video files from this directory
input_video_path = '.'


def preprocess_image(source_image):
    resized_image = cv2.resize(source_image, (NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT))

    # trasnform values from range 0-255 to range -1.0 - 1.0
    resized_image = resized_image - 127.5
    resized_image = resized_image * 0.007843
    return resized_image


def handle_keys(raw_key):
    global min_score_percent
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False
    elif (ascii_code == ord('B')):
        min_score_percent += 5
        print('New minimum box percentage: ' + str(min_score_percent) + '%')
    elif (ascii_code == ord('b')):
        min_score_percent -= 5
        print('New minimum box percentage: ' + str(min_score_percent) + '%')

    return True

def overlay_on_image(display_image, object_info):
    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    base_index = 0
    class_id = object_info[base_index + 1]
    percentage = int(object_info[base_index + 2] * 100)
    if (percentage <= min_score_percent):
        return

    label_text = labels[int(class_id)] + " (" + str(percentage) + "%)"
    box_left = int(object_info[base_index + 3] * source_image_width)
    box_top = int(object_info[base_index + 4] * source_image_height)
    box_right = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)

    box_color = (255, 128, 0)  # box color
    box_thickness = 2
    cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    scale_max = (100.0 - min_score_percent)
    scaled_prob = (percentage - min_score_percent)
    scale = scaled_prob / scale_max

    # draw the classification label string just above and to the left of the rectangle
    #label_background_color = (70, 120, 70)  # greyish green background for text
    label_background_color = (0, int(scale * 175), 75)
    label_text_color = (255, 255, 255)  # white text

    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = box_left
    label_top = box_top - label_size[1]
    if (label_top < 1):
        label_top = 1
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1]
    cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                  label_background_color, -1)

    # label text above the box
    cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

    # display text to let user know how to quit
    cv2.rectangle(display_image,(0, 0),(100, 15), (128, 128, 128), -1)
    cv2.putText(display_image, "Q to Quit", (10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)


def handle_args():
    global resize_output, resize_output_width, resize_output_height
    for an_arg in argv:
        if (an_arg == argv[0]):
            continue

        elif (str(an_arg).lower() == 'help'):
            return False

        elif (str(an_arg).startswith('resize_window=')):
            try:
                arg, val = str(an_arg).split('=', 1)
                width_height = str(val).split('x', 1)
                resize_output_width = int(width_height[0])
                resize_output_height = int(width_height[1])
                resize_output = True
                print ('GUI window resize now on: \n  width = ' +
                       str(resize_output_width) +
                       '\n  height = ' + str(resize_output_height))
            except:
                print('Error with resize_window argument: "' + an_arg + '"')
                return False
        else:
            return False

    return True

def chessboard(img):
	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = numpy.zeros((6*7,3), numpy.float32)
	objp[:,:2] = numpy.mgrid[0:7,0:6].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.

	#images = glob.glob('*.png')


	#img = cv2.imread('template.png')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# Find the chess board corners
	ret, corners = cv2.findChessboardCorners(gray, (5,5),None)

	# If found, add object points, image points (after refining them)
	if ret == True:
		objpoints.append(objp)

		#corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
		#imgpoints.append(corners2)

		# Draw and display the corners
		img = cv2.drawChessboardCorners(img, (7,7), corners,ret)
		#cv2.imshow('img',img)
		#cv2.waitKey(500)

	#cv2.destroyAllWindows()

def run_camera_calibration(cap, nrows, ncols, dimension):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, dimension, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = numpy.zeros((nrows * ncols, 3), numpy.float32)
    objp[:, :2] = numpy.mgrid[0:ncols, 0:nrows].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    ii = 0
    while True:
        ret, img = cap.read()
        if (not ret):
            end_time = time.time()
            print("No image from from video device, exiting")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nrows, ncols), None)
        cv2.imshow(cv_window_name, gray)
        if ret == True:
            ii +=1
            print("cornners found :", ii)
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nrows, ncols), corners2, ret)
            # cv2.imshow('chessboard corners', img)
            cv2.imshow(cv_window_name, img)
            cv2.waitKey(500)
        else:
            print("cornners not found")
            # TODO: when we can controll the car, we can do this automatically by adjust the distance to the front vehicle
        choice = input('Collect more object points, enter  N to stop, press enter to continue:')
        if choice.upper() == 'N':
            break;

    cv2.destroyAllWindows()

    #return ret, camera matrix, distortion coefficients, rotation and translation vectors
    return  cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

	
def run_inference(img, graphnet):

    # preprocess the image to meet nework expectations
    resized_image = preprocess_image(img)

    graphnet.LoadTensor(resized_image.astype(numpy.float16), None)

    output, userobj = graphnet.GetResult()

    #   a.	First fp16 value holds the number of valid detections = num_valid.
    #   b.	The next 6 values are unused.
    #   c.	The next (7 * num_valid) values contain the valid detections data
    #       Each group of 7 values will describe an object/box These 7 values in order.
    #       The values are:
    #         0: image_id (always 0)
    #         1: class_id (this is an index into labels)
    #         2: score (this is the probability for the class)
    #         3: box left location within image as number between 0.0 and 1.0
    #         4: box top location within image as number between 0.0 and 1.0
    #         5: box right location within image as number between 0.0 and 1.0
    #         6: box bottom location within image as number between 0.0 and 1.0

    # number of boxes returned
    num_valid_boxes = int(output[0])

    for box_index in range(num_valid_boxes):
            base_index = 7+ box_index * 7
            if (not numpy.isfinite(output[base_index]) or
                    not numpy.isfinite(output[base_index + 1]) or
                    not numpy.isfinite(output[base_index + 2]) or
                    not numpy.isfinite(output[base_index + 3]) or
                    not numpy.isfinite(output[base_index + 4]) or
                    not numpy.isfinite(output[base_index + 5]) or
                    not numpy.isfinite(output[base_index + 6])):
                # boxes with non finite (inf, nan, etc) numbers must be ignored
                continue

            x1 = max(int(output[base_index + 3] * img.shape[0]), 0)
            y1 = max(int(output[base_index + 4] * img.shape[1]), 0)
            x2 = min(int(output[base_index + 5] * img.shape[0]), img.shape[0]-1)
            y2 = min((output[base_index + 6] * img.shape[1]), img.shape[1]-1)

            # overlay boxes and labels on to the image
            overlay_on_image(img, output[base_index:base_index + 7])


# prints usage information
def print_usage():
    print('\nusage: ')
    print('python3 run_video.py [help][resize_window=<width>x<height>]')
    print('')
    print('options:')
    print('  help - prints this message')
    print('  resize_window - resizes the GUI window to specified dimensions')
    print('                  must be formated similar to resize_window=1280x720')
    print('')
    print('Example: ')
    print('python3 run_video.py resize_window=1920x1080')


def main():
    global resize_output, resize_output_width, resize_output_height

    if (not handle_args()):
        print_usage()
        return 1

    mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)


    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No devices found')
        quit()

    device = mvnc.Device(devices[0])
    device.OpenDevice()

    graph_filename = 'graph'
    with open(graph_filename, mode='rb') as f:
        graph_data = f.read()


    graphnet = device.AllocateGraph(graph_data)



    # template = cv2.imread('template.png',0)
    # template = cv2.resize(template, (NETWORK_IMAGE_WIDTH, NETWORK_IMAGE_HEIGHT))
    # orb = cv2.ORB_create()
    # kp1, des1 = orb.detectAndCompute(template,None)
    cv2.namedWindow(cv_window_name)
    cv2.moveWindow(cv_window_name, 10,  10)

    exit_app = False

    cap = cv2.VideoCapture(0)

    actual_frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print ('actual video resolution: ' + str(actual_frame_width) + ' x ' + str(actual_frame_height))

    if ((cap == None) or (not cap.isOpened())):
        print ('Could not open video device. ')
        exit_app = True


    frame_count = 0
    start_time = time.time()
    end_time = start_time

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ##  Calibration Matrix:
    ##[[ 516.14758188    0.          314.02546443]
    ## [   0.          515.76615942  250.15817809]
    ## [   0.            0.            1.        ]]
    ##Disortion:  [[  2.48041485e-01  -6.31759025e-01   4.36060601e-04  -1.48720850e-03
    ##    5.17810257e-01]]
    ##total error:  0.021112667972552
    mtx = numpy.matrix([[516.14758188, 0 , 314.02546443], [0 , 515.76615942 , 250.15817809], [0, 0, 1]])
    disto = numpy.matrix([[2.48041485e-01,  -6.31759025e-01 ,  4.36060601e-04, -1.48720850e-03, 5.17810257e-01]])

    nrows = 7
    ncols = 7
    dimension = 9

    choice = input('Enter Y to run camera calibration, press enter to continue:')
    if choice.upper() == 'Y':
        ret, mtx, disto, rvecs, tvecs = run_camera_calibration(cap,nrows, ncols, dimension)
        if not ret:
            print('failed to calibrate')
            exit_app = True

    print('mtx',mtx)
    print('disto', disto)
    ret, img = cap.read()
    h, w = img.shape[:-1]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, disto, (w, h), 1, (w, h))

    #mapx, mapy = cv2.initUndistortRectifyMap(mtx, disto, None, newcameramtx, (w, h), 5)

    while(True):
        if (exit_app):
            break
        ret, image = cap.read()
        if (not ret):
            end_time = time.time()
            print("No image from from video device, exiting")
            break

        display_imagec = cv2.undistort(image, mtx, disto, None, newcameramtx)
        x, y, w, h = roi
        display_imagec = display_imagec[y:y + h, x:x + w]

        # check if user hasn't closed the window
        prop_val = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
        if (prop_val < 0.0):
            end_time = time.time()
            exit_app = True
            break

##################################################################################################
        display_image = cv2.cvtColor(display_imagec,cv2.COLOR_BGR2GRAY)
        #corners = cv2.goodFeaturesToTrack(display_image,81, 0.1, 10)

        ret, corners = cv2.findChessboardCorners(display_image, (ncols, nrows), None)

        if ret != True :
            print('No much found')
            continue

        try:
            corners = sorted(numpy.int0(corners).tolist(), key = lambda x:x)
        except:
            continue
        xp = int(actual_frame_width/2)
        yp = int(actual_frame_height/2)
        cv2.circle(display_image, (xp, yp), 10, [150,0,0], -1)
        ii = 0;
        print('######################################')

        sstart = True
        udx = 0
        for corner in corners:
            [x, y] = corner[0]
            if sstart:
                corner[0].append(0)
                xp = x
                yp = y
                sstart = False
                continue
            udx += (x - xp)
            corner[0].append(x - xp)
            xp = x
            yp = y
        print('###################################### udx : ', udx, cnt, len(corners))
        udx /=len(corners)

        print('###################################### udx : ',udx )

        ii = 0
        jj = 0
        sstart = True

        cpx = [[[]]]
        for corner in corners:
            if sstart :
                sstart = False
                continue
            if corner[0][2] < udx :
                cpx[ii].append(corner[0])
                [x, y, dx] = corner[0]
                print(ii, ',', x, ',', y, ',', dx)
            else :
                ii +=1
                cpx.append([corner[0]])
                [x, y, dx] = corner[0]
                print(ii, ',', x, ',', y, ',', dx)

            cv2.circle(display_image,(x,y),3,2000,-1)
        print('######################################')

###################################################################################################
        # kp2, des2 = orb.detectAndCompute(display_imagec,None)
        # matches = bf.match(des1,des2)
        # matches = sorted(matches, key = lambda x:x.distance)
        # display_image = cv2.drawMatches(template,kp1,display_imagec,kp2,matches[:36],None, flags=4)
####################################################################################################
        #run_inference(display_image, graphnet)
        #if (resize_output):
        #display_image = cv2.resize(display_image,(resize_output_width, resize_output_height), cv2.INTER_LINEAR)
####################################################################################################
        cv2.imshow(cv_window_name, display_image)

        raw_key = cv2.waitKey(1)
        if (raw_key != -1):
            if (handle_keys(raw_key) == False):
                end_time = time.time()
                exit_app = True
                break
        frame_count += 1


    frames_per_second = frame_count / (end_time - start_time)
    print('Frames per Second: ' + str(frames_per_second))

    cap.release()

    # Clean up the graph and the device
    graphnet.DeallocateGraph()
    device.CloseDevice()


    cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(main())
