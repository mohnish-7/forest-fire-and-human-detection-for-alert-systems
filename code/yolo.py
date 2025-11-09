def human_presence(net, image, LABELS):

	# importing the necessary packages
	import numpy as np
	import time
	import cv2
	import os

	(H, W) = image.shape[:2]

	# determine only the *output* layer names that we need from YOLO
	laynam = net.getLayerNames()
	ln = [laynam[i-1] for i in net.getUnconnectedOutLayers()]

	# construct a blob from the input image and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	# initialize our lists of detected class IDs
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > 0.5:
				classIDs.append(classID)


	objs = [LABELS[i] for i in classIDs]
	if 'person' in objs:
		return True
	else:
		return False
