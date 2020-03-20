import numpy as np
import argparse
import cv2 as cv
import subprocess
import time
import os
import pyttsx3

from tkinter import *
#def portal():
    
    #window = Tk()
    #window.geometry('900x600')
    #window.title("object detection")
    #image = Image.open('gray.jpg')
    #image = image.resize((900, 600))
    #photo_image = ImageTk.PhotoImage(image)
    #label = Label(window, image = photo_image)
    #label.place(x=0,y=0)
    #btn = Button(window, text="object detection",height=0,fg="black",font=('algerian',20,'bold'),bg="violet",justify='center',command=yolo)
    #btn.place(x=100, y=200)

    #btn1 =Button(window, text="direction",fg="black",font=('algerian',20,'bold'),bg="violet",justify='center',command=object)
    #btn1.place(x=500, y=200)
    #window.mainloop()

def yolo():
    import numpy as np
    import argparse
    import cv2 as cv
    import subprocess
    import time
    import os
    
    

    from yolo_utils import infer_image, show_image
    engine = pyttsx3.init()
    engine.say("Welcome to the Navigation assistant.")
    engine.runAndWait()
    
    FLAGS = []
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model-path',
		type=str,
		default='./yolov3-coco/',
		help='The directory where the model weights and \
			  configuration files are.')

    parser.add_argument('-w', '--weights',
		type=str,
		default='./yolov3-coco/yolov3.weights',
		help='Path to the file which contains the weights \
			 	for YOLOv3.')

    parser.add_argument('-cfg', '--config',
		type=str,
		default='./yolov3-coco/yolov3.cfg',
		help='Path to the configuration file for the YOLOv3 model.')

    parser.add_argument('-i', '--image-path',
		type=str,
		help='The path to the image file')

    parser.add_argument('-v', '--video-path',
		type=str,
		help='The path to the video file')


    parser.add_argument('-vo', '--video-output-path',
		type=str,
        default='./output.avi',
		help='The path of the output video file')

    parser.add_argument('-l', '--labels',
		type=str,
		default='./yolov3-coco/coco-labels',
		help='Path to the file having the \
					labels in a new-line seperated way.')

    parser.add_argument('-c', '--confidence',
		type=float,
		default=0.5,
		help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

    parser.add_argument('-th', '--threshold',
		type=float,
		default=0.3,
		help='The threshold to use when applying the \
				Non-Max Suppresion')

    parser.add_argument('--download-model',
		type=bool,
		default=False,
		help='Set to True, if the model weights and configurations \
				are not present on your local machine.')

    parser.add_argument('-t', '--show-time',
		type=bool,
		default=False,
		help='Show the time taken to infer each image.')

    FLAGS, unparsed = parser.parse_known_args()

	# Download the YOLOv3 models if needed
    if FLAGS.download_model:
	    subprocess.call(['./yolov3-coco/get_model.sh'])

	# Get the labels
    labels = open(FLAGS.labels).read().strip().split('\n')

	# Intializing colors to represent each label uniquely
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model
    net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

	# Get the output layer names of the model
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
	# If both image and video files are given then raise error
    if FLAGS.image_path is None and FLAGS.video_path is None:
        print ('Neither path to an image or path to video provided')
        print ('Starting Inference on Webcam')
        count = 0
        vid = cv.VideoCapture(0)
        while True:
            _, frame = vid.read()
            height, width = frame.shape[:2]
            if count == 0:
                frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
		    						height, width, frame, colors, labels, FLAGS)
                count += 1
            else:
                frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
		    						height, width, frame, colors, labels, FLAGS, boxes, confidences, classids, idxs, infer=False)
                count = (count + 1) % 6
                cv.imshow('webcam', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
        cv.destroyAllWindows()



    
        
def object1():
    import cv2
    import numpy as np
    from scipy.stats import itemfreq


    def get_dominant_color(image, n_colors):
        pixels = np.float32(image).reshape((-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        flags, labels, centroids = cv2.kmeans(
        pixels, n_colors, None, criteria, 10, flags)
        palette = np.uint8(centroids)
        return palette[np.argmax(itemfreq(labels)[:, -1])]


    clicked = False
    def onMouse(event, x, y, flags, param):
        global clicked
        if event == cv2.EVENT_LBUTTONUP:
            clicked = True


    cameraCapture = cv2.VideoCapture(0) 
    cv2.namedWindow('camera')
    cv2.setMouseCallback('camera', onMouse)

    # Read and process frames in loop
    success, frame = cameraCapture.read()




    while success and not clicked:
        cv2.waitKey(1)
        success, frame = cameraCapture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(gray, 37)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
                              1, 50, param1=120, param2=40)

        if not circles is None:
            circles = np.uint16(np.around(circles))
            max_r, max_i = 0, 0
            for i in range(len(circles[:, :, 2][0])):
                if circles[:, :, 2][0][i] > 50 and circles[:, :, 2][0][i] > max_r:
                    max_i = i
                    max_r = circles[:, :, 2][0][i]
            x, y, r = circles[:, :, :][0][max_i]
            if y > r and x > r:
                square = frame[y-r:y+r, x-r:x+r]

                dominant_color = get_dominant_color(square, 2)
                if dominant_color[2] > 100:
                    print("STOP")
                elif dominant_color[0] > 80:
                    zone_0 = square[square.shape[0]*3//8:square.shape[0]
                                    * 5//8, square.shape[1]*1//8:square.shape[1]*3//8]
                    cv2.imshow('Zone0', zone_0)
                    zone_0_color = get_dominant_color(zone_0, 1)

                    zone_1 = square[square.shape[0]*1//8:square.shape[0]
                                * 3//8, square.shape[1]*3//8:square.shape[1]*5//8]
                    cv2.imshow('Zone1', zone_1)
                    zone_1_color = get_dominant_color(zone_1, 1)

                    zone_2 = square[square.shape[0]*3//8:square.shape[0]
                                * 5//8, square.shape[1]*5//8:square.shape[1]*7//8]
                    cv2.imshow('Zone2', zone_2)
                    zone_2_color = get_dominant_color(zone_2, 1)

                    if zone_1_color[2] < 60:
                        if sum(zone_0_color) > sum(zone_2_color):
                            print("LEFT")
                        else:
                            print("RIGHT")
                    else:
                        if sum(zone_1_color) > sum(zone_0_color) and sum(zone_1_color) > sum(zone_2_color):
                            print("FORWARD")
                        elif sum(zone_0_color) > sum(zone_2_color):
                            print("FORWARD AND LEFT")
                        else:
                            print("FORWARD AND RIGHT")
                else:
                    print("N/A")

            for i in circles[0, :]:
                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
        cv2.imshow('camera', frame)


    cv2.destroyAllWindows()
    cameraCapture.release()
yolo()
object1()
