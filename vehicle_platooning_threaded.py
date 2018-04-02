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
import queue
import threading


# name of the opencv window
cv_window_name = "vehicle platooning"

queue_timeout = 0.300

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

	
def render_task():
    global _img_output_queue, resize_output, exit_app
    while (not exit_app):
        if _img_output_queue.empty():
            print('render> img output queue empty')
            time.sleep(0.05)
            continue
        try:
            show_time = time.time()
            (display_image,results) = _img_output_queue.get(True, queue_timeout)
			
            if (resize_output):
                display_image = cv2.resize(display_image,(resize_output_width, resize_output_height), cv2.INTER_LINEAR)
            cv2.imshow(cv_window_name, display_image)
            print('show delay :', time.time()-show_time)
            _img_output_queue.task_done()
        except queue.Empty:
            print('render> except img output queue empty')
            continue
			
def run_inference(img, graphnet):
    image = preprocess_image(img)
    graphnet.LoadTensor(image.astype(numpy.float16), None)

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

            #x1 = max(int(output[base_index + 3] * img.shape[0]), 0)
            #y1 = max(int(output[base_index + 4] * img.shape[1]), 0)
            #x2 = min(int(output[base_index + 5] * img.shape[0]), img.shape[0]-1)
            #y2 = min((output[base_index + 6] * img.shape[1]), img.shape[1]-1)

            # overlay boxes and labels on to the image
            overlay_on_image(img, output[base_index:base_index + 7])

def inference_task(graphnet):
    global _img_input_queue, _img_output_queue, exit_app, render_thread
    start_rendering = True
    while (not exit_app):
        if _img_input_queue.empty():
            print('render> img input queue empty')
            time.sleep(0.05)
            continue
        try:
            inference_time = time.time()
            display_image = _img_input_queue.get(True, queue_timeout)
            run_inference(display_image, graphnet)
            print('inference delay :', time.time()-inference_time)
            _img_output_queue.put((display_image, None), True, queue_timeout)
            if start_rendering :
                render_thread.start()
                start_rendering = False
			
            _img_input_queue.task_done()
        except queue.Empty:
            print('render> except img output queue empty')
            continue
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
    global resize_output, resize_output_width, resize_output_height, _img_input_queue, _img_output_queue, exit_app, render_thread

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
    _img_input_queue = queue.Queue(100)
    _img_output_queue = queue.Queue(50)
    render_thread = threading.Thread(target=render_task, args=())
    inference_thread = threading.Thread(target=inference_task, args=(graphnet,))
    start_inference = True

    while(True):
        if (exit_app):
            break
        photo_time = time.time()
        ret, display_image = cap.read()
        print('camera FPS: ', cap.get(cv2.CAP_PROP_FPS))
        if (not ret):
            end_time = time.time()
            print("No image from from video device, exiting")
            break

		# check if user hasn't closed the window 
        prop_val = cv2.getWindowProperty(cv_window_name, cv2.WND_PROP_ASPECT_RATIO)
        if (prop_val < 0.0):
            end_time = time.time()
            exit_app = True
            break
	    # preprocess the image to meet nework expectations
        #image = preprocess_image(display_image)
        _img_input_queue.put(display_image, True, queue_timeout)
        #run_inference(display_image, graphnet)
        
		
        show_time = time.time()
        print('display time: ', show_time - photo_time)
        if start_inference :
            print('inference_task start ')
            inference_thread.start()
            start_inference = False
			
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
