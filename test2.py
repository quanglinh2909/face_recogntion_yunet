import cv2

from utils.DetectFace import DetectFace
from utils.align_trans import Face_alignment
from utils.face_recognition import FaceRecognition
from utils.facebank import load_facebank

detectFace = DetectFace()
frame = cv2.imread("data/frame.jpg")
bboxes, landmarks = detectFace.detect(frame, w_scale=320, h_scale=240)
# bboxes, landmarks = detectFace.detect(frame)
faceRecognition = FaceRecognition(is_draw=True, threshold=80, tta=True)
faces = Face_alignment(frame, default_square=True, landmarks=landmarks)
                # faces = crop_image(frame, bboxes)
targets, names = load_facebank(path='facebank')

for item in faces:
    cv2.imshow('face', item)
frame, names_result = faceRecognition.recognition(targets=targets, names=names, faces=faces, bboxes=bboxes,
                                           landmarks=landmarks, frame=frame,one_face=True)
print(names_result)

# for item in bboxes:
#     w = item[2] - item[0]
#     h = item[3] - item[1]
#     x = item[0]
#     y = item[1]
#     # cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
#
#     # tăng kích thước bbox
#     x = x - w // 4
#     y = y - h // 4
#     w = w + w // 2
#     h = h + h // 2
#     # cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
#
#     # save face
#     # cv2.imwrite("face.jpg", frame[int(y):int(y + h), int(x):int(x + w)])
#
# for item in landmarks:
#     for i in range(5):
#         cv2.circle(frame, (int(item[i]), int(item[i + 5])), 2, (0, 255, 0), 2)

for item in bboxes:
    cv2.rectangle(frame, (int(item[0]), int(item[1])), (int(item[2]), int(item[3]),), (255, 0, 0), 2)


for item in landmarks:
    for i in range(5):
        cv2.circle(frame, (int(item[i]), int(item[i + 5])), 2, (255, 0, 0), 2)

cv2.imshow('frame', frame)
cv2.waitKey(0)