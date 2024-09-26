import time

import numpy as np

from utils.DetectFace import DetectFace
from utils.align_trans import Face_alignment
import cv2
from torchvision import transforms as trans
import torch
from PIL import Image, ImageDraw, ImageFont

from utils.face_model import MobileFaceNet, l2_norm


class FaceRecognition:

    def __init__(self, modelPath='weights/MobileFace_Net', threshold=0.6, tta=False, is_draw=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detect_model = MobileFaceNet(512).to(self.device)  # embeding size is 512 (feature vector)
        self.detect_model.load_state_dict(
            torch.load(modelPath, map_location=lambda storage, loc: storage, weights_only=False))
        print('MobileFaceNet face detection model generated')
        self.detect_model.eval()
        self.threshold = threshold
        self.tta = tta
        self.is_draw = is_draw
    def get_model(self):
        return self.detect_model

    def recognition(self, targets, names, faces, bboxes, landmarks, frame=None, start_time=None,one_face=False):
        try:
            names_result = []
            if start_time is None:
                start_time = time.time()
            embs = []
            test_transform = trans.Compose([
                trans.ToTensor(),
                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

            for img in faces:
                if self.tta:
                    mirror = cv2.flip(img, 1)
                    emb = self.detect_model(test_transform(img).to(self.device).unsqueeze(0))
                    emb_mirror = self.detect_model(test_transform(mirror).to(self.device).unsqueeze(0))
                    embs.append(l2_norm(emb + emb_mirror))
                else:
                    embs.append(self.detect_model(test_transform(img).to(self.device).unsqueeze(0)))
                if one_face:
                    break

            source_embs = torch.cat(embs)  # number of detected faces x 512
            diff = source_embs.unsqueeze(-1) - targets.transpose(1, 0).unsqueeze(
                0)  # i.e. 3 x 512 x 1 - 1 x 512 x 2 = 3 x 512 x 2
            dist = torch.sum(torch.pow(diff, 2), dim=1)  # number of detected faces x numer of target faces
            minimum, min_idx = torch.min(dist, dim=1)  # min and idx for each row
            min_idx[minimum > ((self.threshold - 156) / (-80))] = -1  # if no match, set idx to -1
            score = minimum
            results = min_idx

            # convert distance to score dis(0.7,1.2) to score(100,60)
            score_100 = torch.clamp(score * -80 + 156, 0, 100)

            if self.is_draw:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image)
                font = ImageFont.truetype('utils/simkai.ttf', 30)

                FPS = 1.0 / (time.time() - start_time)
                draw.text((10, 10), 'FPS: {:.1f}'.format(FPS), fill=(0, 0, 0), font=font)
                for i, b in enumerate(bboxes):
                    draw.rectangle([(b[0], b[1]), (b[2], b[3])], outline='blue', width=5)

                    draw.text((int(b[0]), int(b[1] - 25)), names[results[i] + 1] + ' score:{:.0f}'.format(score_100[i]),
                              fill=(255, 255, 0), font=font)
                    names_result.append([names[results[i] + 1], score_100[i]])
                    if one_face:
                        break

                for p in landmarks:
                    for i in range(5):
                        draw.ellipse([(p[i] - 2.0, p[i + 5] - 2.0), (p[i] + 2.0, p[i + 5] + 2.0)], outline='red')

                    if one_face:
                        break

                frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                return frame, names_result
            else:
                for i, b in enumerate(bboxes):
                    names_result.append([names[results[i] + 1], score_100[i]])
                    if one_face:
                        break

                return frame, names_result
        except Exception as e:
            # print("error: ", e)
            return frame, []
