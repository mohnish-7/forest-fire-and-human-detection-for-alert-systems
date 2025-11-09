import cv2
from yolo import human_presence

# paths to the YOLO weights and model configuration
weightsPath = 'yolo-coco\\yolov3.weights'
configPath = 'yolo-coco\\yolov3.cfg'
labelsPath = './yolo-coco/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")
# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
image = cv2.imread('../Test_Data_HD/HF_5.jpg')

status = human_presence(net, image, LABELS)
if status:
    print('\nHUMAN PRESENT\n')
else:
    print('\nNo Human Found\n')
