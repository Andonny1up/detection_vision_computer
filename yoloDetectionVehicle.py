import cv2 as cv
import numpy as np

# cap = cv.VideoCapture(0)
cap = cv.VideoCapture("./resource/bridge.mp4")
wh_t = 320
conf_threshold = 0.5
nms_my_threshold = 0.3

classes_file = './coco.names'
class_names = []
with open(classes_file,'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')

# print(class_names)
# my variable
vehicle_count = 0
# Mas precicion menos velocidad
model_configuration = './yolov3.cfg'
model_weights = './yolov3.weights'

# Mas velocidad menos precicion
# model_configuration = './yolov3-tiny.cfg'
# model_weights = './yolov3-tiny.weights'

# red
net = cv.dnn.readNetFromDarknet(model_configuration,model_weights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

def find_center(x,y,width,height):
    """X: posicion en el eje x
    Y: posicon en el eje y
    Devuelve el punto del centro
    """
    x_1 = width // 2
    y_1 = height // 2
    
    c_x = x+ x_1
    c_y = y+ y_1
    
    return c_x,c_y


def findObjects(outputs,img):
    """"
    img: frame o fotogramas de un video
    Detecta los objetos en videos"""
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > conf_threshold:
                class_name = class_names[classId]
                if class_name in ["car", "bicycle", "truck", "motorbike", "bus"]:
                    w,h = int(det[2]*wT),int(det[3]*hT)
                    x,y = int((det[0]*wT) -w/2), int((det[1]*hT) -h/2)
                    bbox.append([x,y,w,h])
                    classIds.append(classId)
                    confs.append(float(confidence))
        
    # print(len(bbox))
    indices = cv.dnn.NMSBoxes(bbox,confs,conf_threshold,nms_my_threshold)
    print(indices)
    
    for i in indices:
        box = bbox[i]
        x,y,w,h = box[0], box[1],box[2],box[3]
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv.putText(img,f'{class_names[classIds[i]].upper()} {int(confs[i]*100)}%',
                   (x,y-10), cv.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)
                


while True:
    succes, img = cap.read()
    
    blob = cv.dnn.blobFromImage(img,1/255,(wh_t,wh_t),[0,0,0],1,crop=False)
    net.setInput(blob)
    
    layer_names = net.getLayerNames()
    #print(layer_names)
    output_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    #print(output_names)
    
    #print(net.getUnconnectedOutLayers())
    
    
    outputs = net.forward(output_names)
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])
    # print(type(outputs))
    
    findObjects(outputs,img)
    
    cv.imshow("Image",img)
    cv.waitKey(1)