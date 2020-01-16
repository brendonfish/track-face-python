# USAGE
# python detect_faces_video.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
# for use debug mode on video
# python detect_faces_video.py -l 1 --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
# HaarCascade Face Detection OK..
# Camera RTSP Integrated...

# import the necessary packages
from imutils.video import VideoStream
from collections import deque
import numpy as np
import argparse
import imutils
import time
import cv2
import requests
import json
import threading
import uuid
import nmap
import socket
import time
import threading
import os, shutil
import dlib
from objecttracking.centroidtracker import CentroidTracker
from objecttracking.trackableobject import TrackableObject

global bgImgUrl

faceEvents = deque()
MAX_FACE_EVENTS = 10
N_CAPTURE_PER_EVENT = 3
captureCount = 0
prevDetectCnt = 0


CAM_HOST = "Config_Camera_IP"
CAM_ID_PASS = "Config_Camera_Pass"
SERVER_HOST = "Config_API_URL"
SAMPLE_PERIOD = 8
SNAPSHOT_PERIOD = 15
count = 0
snapcount = 0
brokeFrames = 0
faceDetected = False
LOCAL_SEARCH = False
MIN_FACE_SIZE = (15,15)
MAX_FACE_SIZE = (80,80)
ROI = ( 0, 0, 0, 0)
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
CONFIG_PATH = "../../ai-node-front/system.cfg"
IMG_SNAPSHOT_PATH = "/run/shm/snapshot.jpg"
IMG_CAPTURE_PATH = "/run/shm/capture.jpg"
IMG_CAPTURE_FOLDER = "/run/shm/"

SYS_CONFIG = {}
macAddress = ''

class ipcamCapture:
    def __init__(self, URL):
        self.Frame = []
        self.status = False
        self.isstop = False
		
        if __hank_debug__ == 1:
            #self.capture = cv2.VideoCapture("/Users/poioit/Code/ai/faceai/istockphoto-908008352-640_adpp_is.mp4")
            self.capture = cv2.VideoCapture("/Users/poioit/Code/ai/faceai/rec.mp4")
        else:
            self.capture = cv2.VideoCapture(URL)
        if self.capture.isOpened():
            print("Device Opened\n")   
        else:     
            print("Failed to open Device\n")		

    def start(self):
        print('ipcam started!')
        threading.Thread(target=self.queryframe, args=()).start()

    def stop(self):
        self.isstop = True
        print('ipcam stopped!')
   
    def getframe(self):
        return self.Frame
        
    def queryframe(self):
        while (not self.isstop):
            self.status, self.Frame = self.capture.read()

				
        
        #self.capture.release()


def findnth(haystack, needle, n):
	parts= haystack.split(needle, n+1)
	if len(parts)<=n+1:
		return -1
	return len(haystack)-len(parts[-1])-len(needle)

def get_host_ip():
	try:
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		s.connect(('8.8.8.8', 80))
		ip = s.getsockname()[0]
	finally:
		s.close()

	return ip

def get_local_domain():
	localIp = get_host_ip()
	endIdx = findnth(localIp, ".", 2)
	prefix = localIp[0:endIdx+1]+'0'
	return prefix

def read_config():
	cfgFile = open(CONFIG_PATH, 'r')
	cfgStr = cfgFile.read()
	cfgFile.close()
	return cfgStr

def get_mac():
	if __hank_debug__ == 0:
		ifname = 'eth0'
		macAddress=open('/sys/class/net/%s/address' % ifname).read()
		macAddress=macAddress.replace('\n', '')
		return macAddress
	else:
		mac = 'b8:27:eb:bf:35:dc'
		return mac


def sync():
	# Guard Condition
	macAddress = get_mac()
	print('**********Mac:\n'+'===='+macAddress)
	if macAddress=='':
		return

	# send sync message to cloud
	url = "https://"+ SERVER_HOST + "/v1/sync"
	querystring = {"mac":macAddress}
	headers = {
			'content-type': "application/json",
			'cache-control': "no-cache"
	}
	bodyObj = {
		"videoStatus":"ok",
		"localAddr": get_host_ip()
	}
	bodyStr = json.dumps(bodyObj)

	response = requests.request("PUT", url, verify=False, data=bodyStr, headers=headers, params=querystring)
	print("/sync response"+response.text)
	jsonObj = json.loads(response.text)	
	threading.Timer(30, sync).start()
	print('sync...')
	return

def uploadImage(fileToken, filePath):

	url = "https://"+ SERVER_HOST + "/v1/faceImg"
	querystring = {"file_token":fileToken}
	print('file_token:'+ fileToken)
	print('file_path:' + filePath)
	headers = {
			'content-type': "binary/octet-stream",
			'accept-encoding': "gzip",
			'cache-control': "no-cache"
	}

	response = requests.request("PUT", url, verify=False, data="{}", headers=headers, params=querystring)
	print("/faceImg PUT response"+response.text)
	jsonObj = json.loads(response.text)


	# upload to s3 url
	url = jsonObj['Location']
	querystring = {}
	headers = {
			'content-type': "binary/octet-stream",
			'accept-encoding': "gzip",
			'cache-control': "no-cache"
	}

	response = requests.request("PUT", url, verify=False, data=open(filePath, 'rb'), headers=headers, params=querystring)
	print("S3 PUT response"+response.text)

	# get url from /faceImg
	url = "https://"+ SERVER_HOST + "/v1/faceImg"
	querystring = {"file_token":fileToken}
	headers = {
		'content-type': "binary/octet-stream",
		'accept-encoding': "gzip",
		'cache-control': "no-cache"
	}

	response = requests.request("GET", url, params=querystring)
	print("/faceImg GET response"+response.text)


	# Parse image path
	jsonObj = json.loads(response.text)
	imgUrl = jsonObj['Location']

	return imgUrl


def handleFaceEvent(faceEvent):


	if faceEvent['uploadBgImg']:
		bgImgUrl = uploadImage(faceEvent['bgFileToken'], faceEvent['bgFilePath'])

	imgUrl = uploadImage(faceEvent['fileToken'], faceEvent['filePath'])

	# call /faceEvent
	url = "https://"+ SERVER_HOST + "/v1/face_event"
	headers = {
		'content-type': "application/json",
		'cache-control': "no-cache"
	}
	mac = get_mac()
	body = {
			'imgUrl': imgUrl,
			'mac': mac,
			'shouldNotify':faceEvent['shouldNotify'],
			'imgToken': faceEvent['fileToken'],
			'bgImgUrl': bgImgUrl
	}
	bodyStr = json.dumps(body)	
	response = requests.request("POST", url, data=bodyStr, headers=headers)

	# dump debug info
	print('face_event body:\n'+bodyStr)	
	print("/face_event POST response"+response.text)




def resetFaceEvents():
	while len(faceEvents)>0:
		faceEvent = faceEvents.pop()
		print("**************clean face events...")
		os.remove(faceEvent['filePath'])

def faceEventLoop():
	while True:
		#print("facesevent")
		if len(faceEvents)==0:
			time.sleep(0.1)
		else:
			faceEvent = faceEvents.pop()
			handleFaceEvent(faceEvent)
			os.remove(faceEvent['filePath'])




cfgStr = read_config()
SYS_CONFIG = json.loads(cfgStr)
print('sys config:')
print(SYS_CONFIG)

localDomain=get_local_domain()
print(localDomain)


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-l", "--local_debug", type=int, default=0,
	help="set to 1 for local debug mode")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

__hank_debug__ = args["local_debug"]

# if __hank_debug__ == 0:
sync()
threading.Timer(1, faceEventLoop).start()
ifname = 'eth0'
global targetFrame



camHost=''
vs=''

print("[INFO] reading rtsp stream...")

ipcam = ipcamCapture(SYS_CONFIG['streamURI'])

ipcam.start()

time.sleep(1)

if __hank_debug__ != 1:
	if ipcam.getframe() is None:
		print('RTSP Capture Failed !  URI:' + SYS_CONFIG['streamURI'])
	else:
		print('RTSP Capture OK ! URI:'+ SYS_CONFIG['streamURI'])
		camHost = SYS_CONFIG['streamURI']
	
	if camHost=='':
		print("NO Camera Found. Exit...")
		exit()

if __hank_debug__ == 1:
	FRAME_WIDTH = 640
	FRAME_HEIGHT = 360
	#clean up the pictures, we create in the previous round
	folder = './tmp'
	for the_file in os.listdir(folder):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
				#elif os.path.isdir(file_path): shutil.rmtree(file_path)
		except Exception as e:
					print(e)

# instantiate our centroid tracker, then initialize a list to store
# each of our dlib correlation trackers, followed by a dictionary to
# map each unique object ID to a TrackableObject
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)

trackableObjects = {}
W = FRAME_WIDTH
H = FRAME_HEIGHT
totalDown = 0
totalUp = 0


def detectFace():
	global faceDetected
	global captureCount
	global prevDetectCnt
	newFaceDetected = False
	trackers = []

	# resize frame to have a maximum width of 400 pixels
	roi = SYS_CONFIG['roiCfg']
	startX = int(roi['startX']*FRAME_WIDTH/100)
	startY = int(roi['startY']*FRAME_HEIGHT/100)
	endX = int(roi['endX']*FRAME_WIDTH/100)
	endY = int(roi['endY']*FRAME_HEIGHT/100)
	
	roiFrame = targetFrame[startY:endY, startX:endX]
	cropFrame = roiFrame
	height, width = roiFrame.shape[:2]

	if __hank_debug__ != 1:
		cropFrame = imutils.resize(roiFrame, width=int(width/3))

	# OpenCV works best with gray images
	gray_img = cv2.cvtColor(cropFrame, cv2.COLOR_BGR2GRAY)
	cv2.imwrite('./tmp/detectResult.jpg', gray_img)
	
	# Use OpenCV's built-in Haar classifier
	if __hank_debug__:
		haar_classifier = cv2.CascadeClassifier('/home/pi/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml')
	else:
		haar_classifier = cv2.CascadeClassifier('/home/pi/opencv-3.3.0/data/haarcascades/haarcascade_frontalface_alt.xml')
	faces = haar_classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3, minSize=MIN_FACE_SIZE, maxSize=MAX_FACE_SIZE)
	# print(snapcount)
	# print('Number of faces found: {faces}'.format(faces=len(faces)))

	rects = []
	if len(faces)>0:
		trackers = []
		for i in range(len(faces)):
			(startX, startY, width, height) = faces[i]
	    
			# construct a dlib rectangle object from the bounding
			# box coordinates and then start the dlib correlation
			# tracker
			tracker = dlib.correlation_tracker()
			rect = dlib.rectangle(startX, startY, startX+width, startY+height)
			tracker.start_track(gray_img, rect)
			print('size:'+str(width)+'x'+str(height))
			# cv2.rectangle(gray_img, (startX, startY), (startX+width, startY+height), (255, 255, 0), 2)

			# add the tracker to our list of trackers so we can
			# utilize it during skip frames
			trackers.append(tracker)
			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, startX+width, startY+height))
	else:
		# loop over the trackers
		# print('Number of trackers: {trackers}'.format(trackers=len(trackers)))
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			# status = "Tracking"

			# update the tracker and grab the updated position
			tracker.update(gray_img)
			pos = tracker.get_position()

			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))
		
	# use the centroid tracker to associate the (1) old object
	# centroids with (2) the newly computed object centroids
	objects = ct.update(rects)
	
	# loop over the tracked objects
	# Align with rectangles by rectIdx
	rectIdx = 0
	newRects = []
	for (objectID, centroid) in objects.items():
		# check to see if a trackable object exists for the current
		# object ID
		# print("objectID:" + str(objectID))
		# print("centroid:" + str(centroid[0]) + ":" + str(centroid[1]))
		to = trackableObjects.get(objectID, None)

		# if there is no existing trackable object, create one
		if to is None:
			to = TrackableObject(objectID, centroid)
			newFaceDetected = True
			newRects.append(rects[rectIdx])
			print('***new rect')

		# otherwise, there is a trackable object so we can utilize it
		# to determine direction
		else:
			# the difference between the y-coordinate of the *current*
			# centroid and the mean of *previous* centroids will tell
			# us in which direction the object is moving (negative for
			# 'up' and positive for 'down')
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			# check to see if the object has been counted or not
			if not to.counted:
				# if the direction is negative (indicating the object
				# is moving up) AND the centroid is above the center
				# line, count the object
				if direction < 0 and centroid[1] < H // 2:
					#totalUp += 1
					to.counted = True

				# if the direction is positive (indicating the object
				# is moving down) AND the centroid is below the
				# center line, count the object
				elif direction > 0 and centroid[1] > H // 2:
					#totalDown += 1
					to.counted = True

		# store the trackable object in our dictionary
		trackableObjects[objectID] = to

		# draw both the ID of the object and the centroid of the
		# object on the output frame
	# 	text = "ID {}".format(objectID)
	# 	cv2.putText(gray_img, text, (centroid[0] - 10, centroid[1] - 10),
	# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
	# 	cv2.circle(gray_img, (centroid[0], centroid[1]), 4, (0, 255, 255), -1)
		rectIdx += 1            
	
		
	if len(objects) > prevDetectCnt:
		prevDetectCnt = len(objects)
	#else:
	#	return;

	# prepare face event for each newRects, and push into queue
	print('new Rect')
	print(newRects)

	bgFileToken = uuid.uuid4().hex
	bgImgUrl = ""
	uploadBgImg = False
	bgFilePath = IMG_CAPTURE_FOLDER + bgFileToken + '.jpg'
	cv2.imwrite(bgFilePath, targetFrame)

	if len(newRects)>0:
		uploadBgImg = True

	for rect in newRects:

		fileToken = uuid.uuid4().hex
		if __hank_debug__ == 0:	
			capturePath = IMG_CAPTURE_FOLDER + fileToken + '.jpg'
		else:
			capturePath = './tmp/detectResult' + '/' + fileToken + '.jpg'

		# Resample rectange image and
		# save image to ram disk
		startX = rect[0]
		startY = rect[1]
		endX = rect[2]
		endY = rect[3]
		print(startX)
		print(startY)
		print(endX)
		print(endY)
		print('=======')		
		width = endX - startX
		height = endY - startY
		startX = int(max(0, startX-width/2))
		startY = int(max(0, startY-height/2))
		endX = int(min(cropFrame.shape[1], endX+width/2))
		endY = int(min(cropFrame.shape[0], endY+height/2))
		startX = 3*startX
		startY = 3*startY
		endX = 3*endX
		endY = 3*endY
		print(startX)
		print(startY)
		print(endX)
		print(endY)
		newFrame = roiFrame[startY:endY, startX:endX]
		print(newFrame.shape)
		cv2.imwrite(capturePath, newFrame)

		# Enque Face Event
		faceEvent = {
			'fileToken':fileToken,
			'filePath':capturePath,
			'shouldNotify': True,
			'rectangle': rect,
			'bgFileToken' : bgFileToken,
			'bgFilePath' : bgFilePath,
			'uploadBgImg': uploadBgImg
		}

		# manage the queue size
		if len(faceEvents)<MAX_FACE_EVENTS:
			faceEvents.append(faceEvent)
		else:
			resetFaceEvents()



# loop over the frames from the video stream
while True:
	frame = ipcam.getframe()
	snapcount = snapcount +1
	if snapcount%SNAPSHOT_PERIOD==0:
		snapcount = 0
		cv2.imwrite(IMG_SNAPSHOT_PATH, frame)

	targetFrame = frame
	if targetFrame is None:
		brokeFrames=brokeFrames+1
		if brokeFrames>5:
			vs=cv2.VideoCapture(SYS_CONFIG['streamURI'])
			brokeFrames=0
			continue
		else:
			continue

	brokeFrames=0

	# grab the frame from the threaded video stream and 
	detectFace()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()




