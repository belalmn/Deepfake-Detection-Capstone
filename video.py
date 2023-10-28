import pandas as pd
import numpy as np
import constants as c
from IPython.display import Video as IPythonVideo
import cv2 as cv
from face_recognition import *
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm

class Video:
    def __init__(self, metadata, id, sample=True, frameLimit=15):
        self._get_file_path(metadata, id, sample=sample)
        self._load_video_metadata(metadata, id, sample=sample, frameLimit=frameLimit)
        self._load_frames(frameLimit=frameLimit)
        self._get_all_frame_features()
        self._get_all_frame_feature_centers()
        self._get_all_frame_feature_distances()

    ## Public Methods
    def showVideo(self):
        return IPythonVideo(self.file_path, embed=True)
    
    def drawFrame(self, frame_number):
        image = Image.fromarray(self.frames[frame_number].get_frame())
        plt.imshow(image)

    def drawFaceLandmarks(self, frame_number, color=(255, 255, 255), stroke=2):
        frame = self.frames[frame_number]
        image = frame.draw_face_landmarks(color=color, stroke=stroke)
        top, right, bottom, left = frame.get_face_box()
        image = image.crop((left, top, right, bottom))
        plt.imshow(image)
    
    def drawFaceLandmarkDifference(self, frame_number, other_video):
        frame0 = frame0
        frame1 = frame1

        image0 = Image.fromarray(frame0.get_frame())
        landmarks0 = frame0.draw_face_landmarks(color=(255, 0, 0))
        landmarks1 = frame1.draw_face_landmarks(color=(0, 255, 0))

        overlay = Image.blend(landmarks0, landmarks1, alpha=0.5)
        overlay = Image.blend(base, overlay, alpha=0.5)

        # Crop to face
        top0, right0, bottom0, left0 = frame0.get_face_box()
        top1, right1, bottom1, left1 = frame1.get_face_box()
        min_x, min_y = min(left0, left1), min(top0, top1)
        max_x, max_y = max(right0, right1), max(bottom0, bottom1)
        overlay = overlay.crop((min_x, min_y, max_x, max_y))
        plt.imshow(overlay)

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

    def _load_video_metadata(self, metadata, id, sample, frameLimit):
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
        print("Loading frames...")
        for frame_number in tqdm(range(frameLimit)):
            self.frames.append(Frame(self.cap, self, frame_number))

    def _get_all_frame_features(self):
        features = {
            'chin': [],
            'left_eyebrow': [],
            'right_eyebrow': [],
            'nose_bridge': [],
            'nose_tip': [],
            'left_eye': [],
            'right_eye': [],
            'top_lip': [],
            'bottom_lip': []
        }
        for frame in self.frames:
            for feature in features:
                features[feature].append(frame.get_feature(feature))

        self.chin = pd.DataFrame.from_records(features['chin'])
        self.left_eyebrow = pd.DataFrame.from_records(features['left_eyebrow'])
        self.right_eyebrow = pd.DataFrame.from_records(features['right_eyebrow'])
        self.nose_bridge = pd.DataFrame.from_records(features['nose_bridge'])
        self.nose_tip = pd.DataFrame.from_records(features['nose_tip'])
        self.left_eye = pd.DataFrame.from_records(features['left_eye'])
        self.right_eye = pd.DataFrame.from_records(features['right_eye'])
        self.top_lip = pd.DataFrame.from_records(features['top_lip'])
        self.bottom_lip = pd.DataFrame.from_records(features['bottom_lip'])

    def _get_all_frame_feature_centers(self):
        feature_centers = {
            'chin': [],
            'left_eyebrow': [],
            'right_eyebrow': [],
            'nose_bridge': [],
            'nose_tip': [],
            'left_eye': [],
            'right_eye': [],
            'top_lip': [],
            'bottom_lip': []
        }
        for frame in self.frames:
            for feature in feature_centers:
                feature_centers[feature].append(frame.get_feature_center(feature))

        self.chin['center'] = feature_centers['chin']
        self.left_eyebrow['center'] = feature_centers['left_eyebrow']
        self.right_eyebrow['center'] = feature_centers['right_eyebrow']
        self.nose_bridge['center'] = feature_centers['nose_bridge']
        self.nose_tip['center'] = feature_centers['nose_tip']
        self.left_eye['center'] = feature_centers['left_eye']
        self.right_eye['center'] = feature_centers['right_eye']
        self.top_lip['center'] = feature_centers['top_lip']
        self.bottom_lip['center'] = feature_centers['bottom_lip']
    
    def _get_all_frame_feature_distances(self):
        feature_distances = {
            'chin': [None],
            'left_eyebrow': [None],
            'right_eyebrow': [None],
            'nose_bridge': [None],
            'nose_tip': [None],
            'left_eye': [None],
            'right_eye': [None],
            'top_lip': [None],
            'bottom_lip': [None]
        }
        
        for frame_number in range(1, len(self.frames)):
            for feature in feature_distances:
                feature_distances[feature].append(self.frames[frame_number].get_feature_distance(self.frames[frame_number-1], feature))

        self.chin['distance'] = feature_distances['chin']
        self.left_eyebrow['distance'] = feature_distances['left_eyebrow']
        self.right_eyebrow['distance'] = feature_distances['right_eyebrow']
        self.nose_bridge['distance'] = feature_distances['nose_bridge']
        self.nose_tip['distance'] = feature_distances['nose_tip']
        self.left_eye['distance'] = feature_distances['left_eye']
        self.right_eye['distance'] = feature_distances['right_eye']
        self.top_lip['distance'] = feature_distances['top_lip']
        self.bottom_lip['distance'] = feature_distances['bottom_lip']

class Frame:
    def __init__(self, cap, video, frame_number):
        self.cap = cap
        self.video = video
        self.frame_number = frame_number
        self.frame = self._get_frame()
        self._get_face_locations()
        self._get_face_encodings()
        self._get_face_landmarks()

    ## Public Methods
    def get_frame(self):
        return self.frame

    def get_face_box(self):
        if self.face_locations is None:
            return None
        return self.face_locations[0]
    
    def draw_face_landmarks(self, color=(255, 255, 255), stroke=2):
        # Create blank image of same size as frame
        pil_image = Image.fromarray(self.frame)

        # Create a draw object
        d = ImageDraw.Draw(pil_image)
        d.rectangle([(0, 0), (self.video.width, self.video.height)], fill=(0, 0, 0))
        print(self.chin)
        d.line(self.chin, fill=color, width=stroke)
        d.line(self.left_eyebrow, fill=color, width=stroke)
        d.line(self.right_eyebrow, fill=color, width=stroke)
        d.line(self.nose_bridge, fill=color, width=stroke)
        d.line(self.nose_tip, fill=color, width=stroke)
        d.line(self.left_eye, fill=color, width=stroke)
        d.line(self.right_eye, fill=color, width=stroke)
        d.line(self.top_lip, fill=color, width=stroke)
        d.line(self.bottom_lip, fill=color, width=stroke)
        return pil_image
    
    def get_feature(self, feature):
        features = {
            'chin': self.chin,
            'left_eyebrow': self.left_eyebrow,
            'right_eyebrow': self.right_eyebrow,
            'nose_bridge': self.nose_bridge,
            'nose_tip': self.nose_tip,
            'left_eye': self.left_eye,
            'right_eye': self.right_eye,
            'top_lip': self.top_lip,
            'bottom_lip': self.bottom_lip
        }
        return features[feature]
    
    def get_feature_length(self, feature):
        if self.get_feature(feature) is None:
            return 0
        return len(self.get_feature(feature))
    
    def get_feature_center(self, feature):
        if self.get_feature_length(feature) == 0:
            return np.array([0, 0])
        return np.mean(self.get_feature(feature), axis=0)
    
    def get_feature_distance(self, other_frame, feature):
        if self.get_feature_length(feature) == 0 or other_frame.get_feature_length(feature) == 0:
            return -1
        return np.linalg.norm(self.get_feature_center(feature) - other_frame.get_feature_center(feature))
    
    ## Private Methods
    def _get_frame(self):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, self.frame_number)
        ret, frame = self.cap.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return frame
    
    def _get_face_locations(self):
        faces = face_locations(self.frame)
        num_faces = len(faces)
        self.face_locations = None if num_faces == 0 else faces
        self.num_faces = num_faces
    
    def _get_face_encodings(self):
        if self.face_locations is None:
            self.face_encodings = None
            return
        self.face_encodings = face_encodings(self.frame, self.face_locations)
    
    def _get_face_landmarks(self):
        face = face_landmarks(self.frame, self.face_locations)
        if self.face_locations is None:
            self.chin = None
            self.left_eyebrow = None
            self.right_eyebrow = None
            self.nose_bridge = None
            self.nose_tip = None
            self.left_eye = None
            self.right_eye = None
            self.top_lip = None
            self.bottom_lip = None
        else:
            self.chin = face[0]["chin"]
            self.left_eyebrow = face[0]["left_eyebrow"]
            self.right_eyebrow = face[0]["right_eyebrow"]
            self.nose_bridge = face[0]["nose_bridge"]
            self.nose_tip = face[0]["nose_tip"]
            self.left_eye = face[0]["left_eye"]
            self.right_eye = face[0]["right_eye"]
            self.top_lip = face[0]["top_lip"]
            self.bottom_lip = face[0]["bottom_lip"]