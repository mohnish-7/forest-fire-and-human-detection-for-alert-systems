import cv2
import time
import numpy as np
import os
from os import system
import smtplib
from email.message import EmailMessage
import imghdr
import math
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from pathlib import Path
import torch
import pytorch_lightning as pl
from transformers import ViTFeatureExtractor
from random import shuffle
from twilio.rest import Client
from yolo import human_presence
from coordinates import get_coordinates

# Paths to the YOLO weights and model configuration
weightsPath = 'yolo-coco\\yolov3.weights'
configPath = 'yolo-coco\\yolov3.cfg'
labelsPath = './yolo-coco/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

# Loading the YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Connecting to client for SMS alerts


# Loading the Transformer model
model = torch.load('../Models/new_model.pth')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

# Global variables
frc = 0
frame_counter=0
count = 0
mail_sent = False
hd_alert_sent = False


# ---------------------------------------------------------- Function to Clear the screen --------------------------------------------------- #
def clear():
    system('cls')

# ---------------------------------------------------------- Function to Save an image ------------------------------------------------------ #
def save_img(img,fname):

    dir = r'../Email_Content'
    os.chdir(dir)

    try:
        cv2.imwrite(fname,img)
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Img Captured XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        time.sleep(1.5)
        return True

    except:
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Something Went Wrong while saving the image XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        raise
        # return False



# ------------------------------------------------------- Function to send a mail ----------------------------------------------------------- #

def send_mail(n,content):

    email_address = 'forestfiredetection16@gmail.com'
    password = 'odpxtjtieuobxlgm'

    msg = EmailMessage()
    msg['Subject'] = 'FOREST FIRE ALERT !'
    msg['From'] = email_address
    msg['To'] = email_address
    msg.set_content(content)

    files = ['Local_Fire_Capture_1','Local_Fire_Capture_2','Local_Fire_Capture_3']

    for file in files[:n]:
        with open('../Email_Content/'+file+'.jpg','rb') as f:
            file_data = f.read()
            file_type = imghdr.what(f.name)
            file_name = f.name

        msg.add_attachment(file_data,maintype='image',subtype=file_type,filename=file_name[17:])


    with smtplib.SMTP_SSL('smtp.gmail.com',465) as smtp:
        try:
            print('\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Sending the email XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            smtp.login(email_address,password)
            smtp.send_message(msg)
            return True

        except:
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Something Went Wrong, while sending the email XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

            return False

# ------------------------------------------------------ Function to send SMS Alerts ------------------------------------------------------ #

def send_sms(msg,s=False,w=False):
    if s:
        client.messages.create(to=["+91955489"], from_="+1608738", body= msg)
    if w:
        client.messages.create(from_='whatsapp:+14238886',body=msg,to='whatsapp:+915489')





# ------------------------------------------------------ Function to make a Prediction ------------------------------------------------------ #

def predict(image):

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_prob = logits.softmax(1).tolist()[0][0]
    predicted_class_idx = logits.softmax(1).argmax(-1).item()
    clear()
    if predicted_class_idx == 1:
        return False,0
    else:
        return True,predicted_class_prob



###############################################################################################################################################
#                                                                 Main
###############################################################################################################################################

cap = cv2.VideoCapture('../Video_Data/Fire/Fire_16.wmv')
fps = int(cap.get(cv2.CAP_PROP_FPS))
hd = False

while True and cap.isOpened():


    ret,frame = cap.read()
    time.sleep(1/fps)
    if ret == True:
        frame_counter+=1
        img = frame
        mail_img = img
        img = cv2.resize(img,(250,250))

        if frame_counter%75 == 0:
            fire, prob = predict(img)

            if fire:

                print('FIRE DETECTED !')
                p = "{0:.3f}".format(prob*100)
                print(f'Confidence Score: {p} %')
                if count < 3:

                    fname = 'Local_Fire_Capture_'+str(count+1)+'.jpg'

                    if save_img(mail_img,fname):
                        count += 1

                    if count == 3:

                        msg = "ALERT !\n\n\nFOREST FIRE detected in your neighborhood. STAY SAFE !\n\nConfidence Score: "+str(p)+" %"
                        mail_sent = send_mail(count, msg)
                        send_sms(msg,True,True)

                status = human_presence(net, img, LABELS)
                if status:
                    hd = True
                    print('\nHUMAN PRESENT\n')

                    if count == 3 and hd == True and hd_alert_sent == False:

                        ip, lat, long = get_coordinates()
                        url = f'https://www.google.com/maps/@{lat},{long},17z'
                        msg = "ALERT !\n\n\nPeople are TRAPPED in a FOREST FIRE ! \n\nCoordinates: \nLatitude = "+str(lat)+"\nLongitude = "+str(long)+"\n\nLocation: "+url
                        try:
                            send_mail(count, msg)
                            send_sms(msg,True,True)
                            hd_alert_sent = True

                        except:
                            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX Something Went Wrong, while sending sms XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')

                else:
                    print('\nNo Human Found\n')


            else:
                print('No Fire Detected.')



        frame = cv2.resize(frame,(900,600))
        #time.sleep(1/fps)
        cv2.imshow('Video Feed',frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

if not mail_sent and count > 0:
    msg = "ALERT !\n\n\nFOREST FIRE detected in your neighborhood. STAY SAFE !\n\nConfidence Score: "+str(p)+" %"
    send_mail(count, msg)
    send_sms(msg,False,True)
    mail_sent = True


if mail_sent and hd_alert_sent:
    print('\n\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n       ALERTS SENT SUCCESSFULLY       \n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
