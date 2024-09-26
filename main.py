import time

import cv2

from utils.DetectFace import DetectFace
from utils.align_trans import Face_alignment, crop_image
from utils.face_recognition import FaceRecognition
from utils.facebank import load_facebank, prepare_facebank

# cap = cv2.VideoCapture("data/output_faces_video.mp4")
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("rtsp://admin:Oryza123@192.168.104.189:554/cam/realmonitor?channel=1&subtype=0")
# cap = cv2.VideoCapture("rtsp://admin:Oryza123@192.168.103.210:554/cam/realmonitor?channel=1&subtype=0")

threshold = 90

detectFace = DetectFace()
faceRecognition = FaceRecognition(is_draw=True,threshold=threshold,tta=True)
targets, names = load_facebank(path='facebank')

# targets, names = prepare_facebank(faceRecognition.get_model(),path='facebank',detectFace=detectFace)
start = time.time()
bboxes = []
listName = []
while True:
    ret, frame = cap.read()
    if ret:
        if time.time() - start > 0.1:
            start = time.time()
            start_time = time.time()
            bboxes, landmarks = detectFace.detect(frame, w_scale=320, h_scale=240)
            listFace = detectFace.tranform_detect(frame,bboxes)
            listName = []
            for item in listFace:
                b2, l2 = detectFace.detect(item)
                name = ""
                if len(b2) > 0:
                    faces = Face_alignment(item, default_square=True, landmarks=l2)
                    _, names_result = faceRecognition.recognition(targets=targets, names=names, faces=faces, bboxes=b2,
                                                           landmarks=l2, frame=item, start_time=start_time,one_face=True)
                    if len(names_result) > 0 and names_result[0][0] !="":
                        print(names_result[0][0])
                        name = names_result[0][0]
                listName.append(name)
                break

        for i in range(len(bboxes)):
            x1, y1, x2, y2,_ = bboxes[i]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2),), (255, 0, 0), 2)
            if i == 0:
                cv2.putText(frame, listName[i], (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
