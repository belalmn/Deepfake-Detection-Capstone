import pandas as pd
import numpy as np
import constants as c
from IPython.display import Video
import cv2 as cv
from face_recognition import *
from PIL import Image, ImageDraw

class Video:
    def __init__(self, metadata, id, sample=True, frameLimit=None):
        self._load_video(metadata, id, sample=sample, frameLimit=frameLimit)

    ## Public Methods
    def showVideo(self):
        return Video(self.file_path, embed=True)

    ## Private Methods
    def _get_file_path(self, metadata, id, sample):
        folder_path = c.SAMPLE_DATA_DIR if sample else c.FULL_DATA_DIR
        file_path = folder_path
        file_name = '/' + str(id) + '.mp4'
        if sample:
            file_path += file_name
        else:
            file_path += '/' + metadata[id]['folder'] + file_name
        self.file_path = file_path

    def _load_video(self, metadata, id, sample, frameLimit):
        # Get file path
        self._get_file_path(metadata, id, sample=sample)

        # Load video metadata
        self.cap = cv.VideoCapture(self.file_path)
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count/self.fps
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.id = id
    
    def _load_frames(self, frameLimit):
        self.frames = []
        frameLimit = self.frame_count if frameLimit is None else frameLimit
        for frame_number in range(frameLimit):
            self.frames.append(Frame(self.cap, self, frame_number))
    
class Frame:
    def __init__(self, cap, video, frame_number):
        self.cap = cap
        self.video = video
        self.frame_number = frame_number
        self.frame = self._get_frame()
        self.face_locations = self._get_face_locations()
        self.face_encodings = self._get_face_encodings()
        self.face_landmarks = self._get_face_landmarks()

    def _get_frame(self, frame_number):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame
    
    def _get_face_locations(self, frame):
        return face_locations(frame)
    
    def _get_face_encodings(self, frame, face_locations):
        return face_encodings(frame, face_locations)
    
    def _get_face_landmarks(self, frame, face_locations):
        return face_landmarks(frame, face_locations)