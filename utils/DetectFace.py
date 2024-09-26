import cv2
import numpy as np

from utils.yunet import YuNet


class DetectFace:
    def __init__(self, modelPath='weights/face_detection_yunet_2023mar.onnx'):
        self.detector = YuNet(modelPath=modelPath,
                              inputSize=[320, 320],
                              confThreshold=0.7,
                              nmsThreshold=0.3,
                              topK=5000,
                              backendId=cv2.dnn.DNN_BACKEND_OPENCV,
                              targetId=cv2.dnn.DNN_TARGET_CPU)

    def detect(self, img, w_scale=0, h_scale=0):
        image = img.copy()

        bboxs = []
        landmarks = []
        w, h = image.shape[1], image.shape[0]
        if w_scale == 0 and h_scale == 0:
            w_scale = w
            h_scale = h
        else:
            image = cv2.resize(image, (w_scale, h_scale))


        self.detector.setInputSize([w_scale, h_scale])
        results = self.detector.infer(image)
        for det in results:
            bbox = det[0:4].astype(np.int32).reshape(4)
            if w_scale != w and h_scale != h:
                bbox[0] = int(bbox[0] * w / w_scale)
                bbox[1] = int(bbox[1] * h / h_scale)
                bbox[2] = int(bbox[2] * w / w_scale)
                bbox[3] = int(bbox[3] * h / h_scale)

            conf = det[-1]
            box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3], conf]
            land = det[4:14].astype(np.int32).reshape(5, 2)
            l1 = []
            l2 = []
            for idx, landmark in enumerate(land):
                if w_scale != w and h_scale != h:
                    landmark[0] = int(landmark[0] * w / w_scale)
                    landmark[1] = int(landmark[1] * h / h_scale)
                l1.append(landmark[0])
                l2.append(landmark[1])

            landmarks.append(l1 + l2)
            bboxs.append(box)

        landmarks = np.array(landmarks)
        bboxs = np.array(bboxs)
        return bboxs, landmarks

    def tranform_detect(self,frame,bboxes):
        list_faces = []
        for item in bboxes:
            w = item[2] - item[0]
            h = item[3] - item[1]
            x = item[0]
            y = item[1]
            # cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

            # tăng kích thước bbox
            x = x - w // 4
            y = y - h // 4
            w = w + w // 2
            h = h + h // 2
            # cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)

            # save face
            # cv2.imwrite("face.jpg", frame[int(y):int(y + h), int(x):int(x + w)])
            list_faces.append(frame[int(y):int(y + h), int(x):int(x + w)])
        return list_faces


