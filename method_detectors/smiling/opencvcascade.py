"""
Run smile detector from OpenCV.

https://www.geeksforgeeks.org/python-smile-detection-using-opencv/
"""

import os
import cv2
from method_detectors.base_detector import BaseDetector


class OpencvCascade(BaseDetector):
    """Class to setup and run existing computer vision research code.
    """

    name = 'smile'
    algorithm = 'EmoNet'

    def __init__(self, settings) -> None:
        """InitializeMethod class.

        Parameters
        ----------
        settings : dict
            some configurations/settings dictionary
        """
        super().__init__(settings)

        self.cascades_folder = settings['cascades_folder']

        # include haar-cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + os.path.join(self.cascades_folder,
                                                 'haarcascade_frontalface.xml'))
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml')

    def detect(self, gray, frame):
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)),
                              (0, 0, 255), 2)
        return frame

    def inference(self):
        """Run inference of the method on the pre-loaded image

        Returns
        -------
        dict
            dict(method_name:str, values:list)
            the values list contains entries of the form
                dict(feature:str, start:int, end:int, label:str)
        """
        # check that the data was created
        assert self.segments_list is not None, \
            f"{self.name}: Please initialize the data " \
            f"(via 'self.data_initialization()') before running inference."

        detections = dict(name=self.name, values=[])
        for (video_file, start_time, end_time) in self.segments_list:

            video_capture = cv2.VideoCapture(video_file)
            while video_capture.isOpened():
                # Captures video_capture frame by frame
                _, frame = video_capture.read()

                # To capture image in monochrome
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # calls the detect() function
                canvas = self.detect(gray, frame)


            result = 5
            print(result)

            detections['values'].append(dict(
                algorithm=self.algorithm,
                start=start_time,
                end=end_time,
                label=label_names[gesture],
                probability=probability
            ))
        return detections






