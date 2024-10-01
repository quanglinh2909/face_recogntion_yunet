import asyncio
import time

import cv2

from app.websocket.connection_manager import connection_manager
from utils.DetectFace import DetectFace
from utils.align_trans import Face_alignment, crop_image
from utils.face_recognition import FaceRecognition
from utils.facebank import load_facebank, prepare_facebank
import os


def start_detect():
    cap = cv2.VideoCapture("rtsp://admin:Oryza1234@192.168.103.210:554/cam/realmonitor?channel=1&subtype=1")
    threshold = 90

    detectFace = DetectFace()
    faceRecognition = FaceRecognition(is_draw=True, threshold=threshold, tta=True)
    # get path of facebank
    path = 'facebank'
    path_root = os.path.join(os.path.dirname(__file__), path)
    path_root = path_root.replace("utils", "")
    targets, names = load_facebank(path=path_root)

    # targets, names = prepare_facebank(faceRecognition.get_model(),path='facebank',detectFace=detectFace)
    start = time.time()
    bboxes = []
    listName = []
    while True:
        ret, frame = cap.read()

        if ret:
            if time.time() - start > 1:
                start = time.time()
                start_time = time.time()
                bboxes, landmarks = detectFace.detect(frame, w_scale=224, h_scale=192)
                listFace = detectFace.tranform_detect(frame, bboxes)
                listName = []
                for item in listFace:
                    b2, l2 = detectFace.detect(item)
                    name = ""
                    if len(b2) > 0:
                        faces = Face_alignment(item, default_square=True, landmarks=l2)
                        _, names_result = faceRecognition.recognition(targets=targets, names=names, faces=faces,
                                                                      bboxes=b2,
                                                                      landmarks=l2, frame=item, start_time=start_time,
                                                                      one_face=True)
                        if len(names_result) > 0 and names_result[0][0] != "":
                            print(names_result[0][0])
                            name = names_result[0][0]

                            # asyncio.run(
                            #     connection_manager.send_company_message_json(
                            #         company_id="1", message={"data": {"user_id":'664d9619073d2177bfe0ed78', "camera_id": "66ece1edf0e208b33619ab49",
                            #                                           "box_face": [int(b2[0][0]), int(b2[0][1]), int(b2[0][2]), int(b2[0][3])]}}
                            #     )
                            # )
                            camera_id = "66ece1edf0e208b33619ab49"
                            name_slit = name.split("@")
                            id_user = ''
                            name_user = ''
                            if len(name_slit) >= 2:
                                name_user = name_slit[0]
                                id_user = name_slit[1]
                            asyncio.run(
                                connection_manager.send_company_message_json(
                                    company_id="1", message={
                                        "id": "66fb6daf80c6312c3ec70477",
                                        "data": {
                                            "timestamp": time.time(),
                                            "image_url": "",
                                            "box_face": [int(bboxes[0][0]), int(bboxes[0][1]), int(bboxes[0][2]), int(bboxes[0][3])],
                                            "detection_confidence": 0.707,
                                            "company_id": "",
                                            "user_id": id_user,
                                            "name": name_user,
                                            "camera_id": camera_id,
                                            "camera_name": "Bãi xe làn ra",
                                            "camera_ip": "192.168.104.189"
                                        },
                                        "camera": camera_id,
                                        "created": "2024-10-01T10:34:07"
                                    }
                                )
                            )

                    listName.append(name)
                    break

            for i in range(len(bboxes)):
                x1, y1, x2, y2, _ = bboxes[i]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2),), (255, 0, 0), 2)
                if i == 0:
                    cv2.putText(frame, listName[i], (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (36, 255, 12), 2)

        # frame = cv2.resize(frame, (224, 192))
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
