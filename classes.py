import pandas as pd
import numpy as np
import constants as c
from IPython.display import Video as IPythonVideo
import cv2 as cv
from face_recognition import *
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from numba import jit
import json

class Video:
    def __init__(self, metadata, id, frameLimit=150, sample=True, tq="notebook"):
        self._get_file_path(metadata, id, sample=sample)
        self._load_video_metadata()
        self._load_frames(frameLimit=frameLimit, tq=tq)
        self._get_all_frame_features()
        self._get_all_frame_feature_centers()
        self._get_all_frame_feature_distances()
        self._get_all_frame_feature_velocities()

    ## Public Methods
    def showVideo(self):
        """
        Embed full video in Jupyter Notebook

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        return IPythonVideo(self.file_path, embed=True)

    def drawFrame(self, frame_number):
        """
        Display a given frame

        Parameters
        ----------
        frame_number : int
            Frame number to draw

        Returns
        ----------
        None
        """
        image = Image.fromarray(self.frames[frame_number].getFrame())
        plt.imshow(image)

    def drawFaceLandmarks(self, frame_number, color=(255, 255, 255), stroke=2):
        """
        Visualize face landmarks for a given frame

        Parameters
        ----------
        frame_number : int
            Frame number to draw
        color : tuple
            Color of face landmarks
        stroke : int
            Stroke of face landmarks

        Returns
        ----------
        None
        """
        frame: Frame = self.frames[frame_number]
        image = Image.fromarray(frame.getFrame())
        landmarks = frame.drawFaceLandmarks(color=color, stroke=stroke)

        overlay = Image.blend(image, landmarks, alpha=0.5)

        # Crop to face
        top, right, bottom, left = frame.getFaceBox()
        overlay = overlay.crop((left, top, right, bottom))
        plt.imshow(overlay)

    def drawFaceLandmarkComparison(self, frame_number, other_video):
        """
        Visualize comparison of face landmarks with another video

        Parameters
        ----------
        frame_number : int
            Frame number to draw
        other_video : Video
            Video to compare to

        Returns
        ----------
        None
        """
        frame0: Frame = self.frames[frame_number]
        frame1: Frame = other_video.frames[frame_number]

        image0 = Image.fromarray(frame0.getFrame())
        landmarks0 = frame0.drawFaceLandmarks(color=(0, 255, 0))
        landmarks1 = frame1.drawFaceLandmarks(color=(255, 0, 0))

        overlay = Image.blend(landmarks0, landmarks1, alpha=0.5)
        overlay = Image.blend(image0, overlay, alpha=0.5)

        # Crop to face
        top0, right0, bottom0, left0 = frame0.getFaceBox()
        top1, right1, bottom1, left1 = frame1.getFaceBox()
        min_x, min_y = min(left0, left1), min(top0, top1)
        max_x, max_y = max(right0, right1), max(bottom0, bottom1)
        overlay = overlay.crop((min_x, min_y, max_x, max_y))
        plt.imshow(overlay)

    def graphFeatureDistance(self, feature: str):
        """
        Graph the difference in position for a given feature between all frames of a video

        Parameters
        ----------
        feature : str
            Feature to graph

        Returns
        ----------
        None
        """
        feature_df = self.getFeatureDf(feature)
        if feature_df is None:
            print("Invalid feature")
            return
        plt.plot(feature_df["distance"])
        plt.plot(feature_df["distance"], marker="o", markevery=[0])
        plt.xlabel("Frame Number")
        plt.ylabel("Distance")
        plt.title(
            "Distance Between "
            + feature.replace("_", " ").title()
            + " Centers in "
            + self.id
        )
        plt.show()

    def graphFeatureDistanceComparison(self, feature: str, other_video: "Video"):
        """
        Graph the difference in position for a given feature between two videos

        Parameters
        ----------
        feature : str
            Feature to graph
        other_video : Video
            Video to compare to

        Returns
        ----------
        None
        """
        feature_df = self.getFeatureDf(feature)
        other_feature_df = other_video.getFeatureDf(feature)
        if feature_df is None or other_feature_df is None:
            print("Invalid feature")
            return
        plt.plot(feature_df["distance"], color="green")
        plt.plot(other_feature_df["distance"], color="red")
        plt.plot(feature_df["distance"], color="green", marker="o", markevery=[0])
        plt.plot(other_feature_df["distance"], color="red", marker="o", markevery=[0])
        plt.xlabel("Frame Number")
        plt.ylabel("Distance")
        plt.title(
            "Distance Between "
            + feature.replace("_", " ").title()
            + " Centers in "
            + self.id
            + " and "
            + other_video.id
        )
        plt.show()

    def drawFeatureCentralPositions(self, feature: str):
        """
        Draw the positional centers of a given feature for all frames, with connected lines

        Parameters
        ----------
        feature : str
            Feature to graph

        Returns
        ----------
        None
        """
        feature_df = self.getFeatureDf(feature)
        if feature_df is None:
            print("Invalid feature")
            return
        plt.plot(
            feature_df["central_position"].apply(lambda x: x[0]),
            feature_df["central_position"].apply(lambda x: x[1]),
        )
        plt.plot(
            feature_df["central_position"].apply(lambda x: x[0]),
            feature_df["central_position"].apply(lambda x: x[1]),
            marker="o",
            markevery=[0],
        )
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title(
            "Positional Centers of "
            + feature.replace("_", " ").title()
            + " in "
            + self.id
        )
        plt.show()

    def drawFeatureCentralPositionsComparison(self, feature: str, other_video: "Video"):
        """
        Draw the positional centers of a given feature for all frames of two videos, with connected lines

        Parameters
        ----------
        feature : str
            Feature to graph
        other_video : Video
            Video to compare to

        Returns
        ----------
        None
        """
        feature_df = self.getFeatureDf(feature)
        other_feature_df = other_video.getFeatureDf(feature)
        if feature_df is None or other_feature_df is None:
            print("Invalid feature")
            return
        plt.plot(
            feature_df["central_position"].apply(lambda x: x[0]),
            feature_df["central_position"].apply(lambda x: x[1]),
            color="green",
        )
        plt.plot(
            other_feature_df["central_position"].apply(lambda x: x[0]),
            other_feature_df["central_position"].apply(lambda x: x[1]),
            color="red",
        )
        plt.plot(
            feature_df["central_position"].apply(lambda x: x[0]),
            feature_df["central_position"].apply(lambda x: x[1]),
            color="green",
            marker="o",
            markevery=[0],
        )
        plt.plot(
            other_feature_df["central_position"].apply(lambda x: x[0]),
            other_feature_df["central_position"].apply(lambda x: x[1]),
            color="red",
            marker="o",
            markevery=[0],
        )
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.title(
            "Positional Centers of "
            + feature.replace("_", " ").title()
            + " in "
            + self.id
            + " and "
            + other_video.id
        )
        plt.show()

    def graphFeatureVelocity(self, feature: str):
        """
        Graph the velocity of a given feature between all frames of a video

        Parameters
        ----------
        feature : str
            Feature to graph

        Returns
        ----------
        None
        """
        feature_df = self.getFeatureDf(feature)
        if feature_df is None:
            print("Invalid feature")
            return
        plt.plot(feature_df["velocity_x"], label="X Velocity")
        plt.plot(feature_df["velocity_y"], label="Y Velocity")
        plt.plot(feature_df["velocity_x"], marker="o", markevery=[0])
        plt.plot(feature_df["velocity_y"], marker="o", markevery=[0])
        plt.xlabel("Frame Number")
        plt.ylabel("Velocity")
        plt.title(
            "Velocity of "
            + feature.replace("_", " ").title()
            + " in "
            + self.id
        )
        plt.show()

    def graphFeatureVelocityXComparison(self, feature: str, other_video: "Video"):
        """
        Graph the velocity of a given feature between two videos

        Parameters
        ----------
        feature : str
            Feature to graph
        other_video : Video
            Video to compare to

        Returns
        ----------
        None
        """
        feature_df = self.getFeatureDf(feature)
        other_feature_df = other_video.getFeatureDf(feature)
        if feature_df is None or other_feature_df is None:
            print("Invalid feature")
            return
        plt.plot(feature_df["velocity_x"], color="green", label=self.id)
        plt.plot(other_feature_df["velocity_x"], color="red", label=other_video.id)
        plt.plot(feature_df["velocity_x"], color="green", marker="o", markevery=[0])
        plt.plot(other_feature_df["velocity_x"], color="red", marker="o", markevery=[0])
        plt.xlabel("Frame Number")
        plt.ylabel("X Velocity")
        plt.title(
            "X Velocity of "
            + feature.replace("_", " ").title()
            + " in "
            + self.id
            + " and "
            + other_video.id
        )
        plt.show()

    def graphFeatureVelocityYComparison(self, feature: str, other_video: "Video"):
        """
        Graph the velocity of a given feature between two videos

        Parameters
        ----------
        feature : str
            Feature to graph
        other_video : Video
            Video to compare to

        Returns
        ----------
        None
        """
        feature_df = self.getFeatureDf(feature)
        other_feature_df = other_video.getFeatureDf(feature)
        if feature_df is None or other_feature_df is None:
            print("Invalid feature")
            return
        plt.plot(feature_df["velocity_y"], color="green", label=self.id)
        plt.plot(other_feature_df["velocity_y"], color="red", label=other_video.id)
        plt.plot(feature_df["velocity_y"], color="green", marker="o", markevery=[0])
        plt.plot(other_feature_df["velocity_y"], color="red", marker="o", markevery=[0])
        plt.xlabel("Frame Number")
        plt.ylabel("Y Velocity")
        plt.title(
            "Y Velocity of "
            + feature.replace("_", " ").title()
            + " in "
            + self.id
            + " and "
            + other_video.id
        )
        plt.show()
    
    def graphFeatureVelocityComparison(self, feature: str, other_video: "Video"):
        """
        Graph the velocity of a given feature between two videos

        Parameters
        ----------
        feature : str
            Feature to graph
        other_video : Video
            Video to compare to

        Returns
        ----------
        None
        """
        feature_df = self.getFeatureDf(feature)
        other_feature_df = other_video.getFeatureDf(feature)
        if feature_df is None or other_feature_df is None:
            print("Invalid feature")
            return
        plt.plot(feature_df["velocity_x"], color="green")
        plt.plot(feature_df["velocity_y"], color="green")
        plt.plot(other_feature_df["velocity_x"], color="red")
        plt.plot(other_feature_df["velocity_y"], color="red")
        plt.plot(feature_df["velocity_x"], color="green", marker="o", markevery=[0])
        plt.plot(feature_df["velocity_y"], color="green", marker="o", markevery=[0])
        plt.plot(other_feature_df["velocity_x"], color="red", marker="o", markevery=[0])
        plt.plot(other_feature_df["velocity_y"], color="red", marker="o", markevery=[0])
        plt.xlabel("Frame Number")
        plt.ylabel("Velocity")
        plt.title(
            "Velocity of "
            + feature.replace("_", " ").title()
            + " in "
            + self.id
            + " and "
            + other_video.id
        )
        plt.show()
        
    def getFeatureDf(self, feature: str):
        """
        Get a feature dataframe for all frames

        Parameters
        ----------
        feature : str
            Feature to get

        Returns
        ----------
        pandas.DataFrame
            Dataframe of feature
        """
        match feature:
            case "chin":
                return self.chin
            case "left_eyebrow":
                return self.left_eyebrow
            case "right_eyebrow":
                return self.right_eyebrow
            case "nose_bridge":
                return self.nose_bridge
            case "nose_tip":
                return self.nose_tip
            case "left_eye":
                return self.left_eye
            case "right_eye":
                return self.right_eye
            case "top_lip":
                return self.top_lip
            case "bottom_lip":
                return self.bottom_lip
            case _:
                return None

    def getVelocityVariance(self, feature: str):
        """
        Get the variance of a given feature's velocity

        Parameters
        ----------
        feature : str
            Feature to get

        Returns
        ----------
        float
            Variance of feature's velocity
        """
        feature_df = self.getFeatureDf(feature)
        if feature_df is None:
            print("Invalid feature")
            return
        return np.var(feature_df["velocity_x"]) + np.var(feature_df["velocity_y"])
    
    ## Private Methods
    def _get_file_path(self, metadata: "Metadata", id, sample):
        """
        Get file path for video

        Parameters
        ----------
        metadata : Metadata
            Metadata object
        id : str
            Video id
        sample : bool
            True if sample video, False otherwise

        Returns
        ----------
        None
        """
        folder_prefix = c.FULL_DATA_FOLDER_PREFIX
        folder_path = (
            c.SAMPLE_DATA_DIR
            if sample
            else c.FULL_DATA_DIR + folder_prefix + metadata[id]["folder"]
        )
        file_path = folder_path
        file_name = "/" + str(id) + ".mp4"
        if sample:
            file_path += file_name
        else:
            file_path += "/" + metadata[id]["folder"] + file_name
        self.file_path = file_path
        self.id = id

    def _load_video_metadata(self):
        """
        Load metadata for video

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        self.cap = cv.VideoCapture(self.file_path)
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    def _load_frames(self, frameLimit, tq):
        """
        Load frames for video

        Parameters
        ----------
        frameLimit : int
            Number of frames to load
        tqdm : str
            Tqdm notebook or tqdm

        Returns
        ----------
        None
        """
        self.frames: list[Frame] = []
        frameLimit = self.frame_count if frameLimit is None else frameLimit
        _tqdm = tqdm_notebook if tq == "notebook" else tqdm
        print(f"Loading frames for video {self.id}...")
        for frame_number in _tqdm(range(frameLimit)):
            self.frames.append(Frame(self.cap, self, frame_number))

    def _get_all_frame_features(self):
        """
        Get all features for all frames

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        features = {
            "chin": [],
            "left_eyebrow": [],
            "right_eyebrow": [],
            "nose_bridge": [],
            "nose_tip": [],
            "left_eye": [],
            "right_eye": [],
            "top_lip": [],
            "bottom_lip": [],
        }
        for frame in self.frames:
            for feature in features:
                features[feature].append(frame.getFeature(feature))

        self.chin = pd.DataFrame.from_records(features["chin"])
        self.left_eyebrow = pd.DataFrame.from_records(features["left_eyebrow"])
        self.right_eyebrow = pd.DataFrame.from_records(features["right_eyebrow"])
        self.nose_bridge = pd.DataFrame.from_records(features["nose_bridge"])
        self.nose_tip = pd.DataFrame.from_records(features["nose_tip"])
        self.left_eye = pd.DataFrame.from_records(features["left_eye"])
        self.right_eye = pd.DataFrame.from_records(features["right_eye"])
        self.top_lip = pd.DataFrame.from_records(features["top_lip"])
        self.bottom_lip = pd.DataFrame.from_records(features["bottom_lip"])

    def _get_all_frame_feature_centers(self):
        """
        Get every feature's positional center for all frames

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        feature_centers = {
            "chin": [],
            "left_eyebrow": [],
            "right_eyebrow": [],
            "nose_bridge": [],
            "nose_tip": [],
            "left_eye": [],
            "right_eye": [],
            "top_lip": [],
            "bottom_lip": [],
        }
        for frame in self.frames:
            for feature in feature_centers:
                feature_centers[feature].append(
                    frame.getFeatureCentralPosition(feature)
                )

        self.chin["central_position"] = feature_centers["chin"]
        self.left_eyebrow["central_position"] = feature_centers["left_eyebrow"]
        self.right_eyebrow["central_position"] = feature_centers["right_eyebrow"]
        self.nose_bridge["central_position"] = feature_centers["nose_bridge"]
        self.nose_tip["central_position"] = feature_centers["nose_tip"]
        self.left_eye["central_position"] = feature_centers["left_eye"]
        self.right_eye["central_position"] = feature_centers["right_eye"]
        self.top_lip["central_position"] = feature_centers["top_lip"]
        self.bottom_lip["central_position"] = feature_centers["bottom_lip"]

    def _get_all_frame_feature_distances(self):
        """
        Get every feature's average change in distance from the previous frame for all frames

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        feature_distances = {
            "chin": [None],
            "left_eyebrow": [None],
            "right_eyebrow": [None],
            "nose_bridge": [None],
            "nose_tip": [None],
            "left_eye": [None],
            "right_eye": [None],
            "top_lip": [None],
            "bottom_lip": [None],
        }

        for frame_number in range(1, len(self.frames)):
            for feature in feature_distances:
                feature_distances[feature].append(
                    self.frames[frame_number].getFeatureDistance(
                        self.frames[frame_number - 1], feature
                    )
                )

        self.chin["distance"] = feature_distances["chin"]
        self.left_eyebrow["distance"] = feature_distances["left_eyebrow"]
        self.right_eyebrow["distance"] = feature_distances["right_eyebrow"]
        self.nose_bridge["distance"] = feature_distances["nose_bridge"]
        self.nose_tip["distance"] = feature_distances["nose_tip"]
        self.left_eye["distance"] = feature_distances["left_eye"]
        self.right_eye["distance"] = feature_distances["right_eye"]
        self.top_lip["distance"] = feature_distances["top_lip"]
        self.bottom_lip["distance"] = feature_distances["bottom_lip"]

    def _get_all_frame_feature_velocities(self):
        """
        Get every feature's average change in velocity from the previous frame for all frames

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        feature_velocity_x = {
            "chin": [None, None],
            "left_eyebrow": [None, None],
            "right_eyebrow": [None, None],
            "nose_bridge": [None, None],
            "nose_tip": [None, None],
            "left_eye": [None, None],
            "right_eye": [None, None],
            "top_lip": [None, None],
            "bottom_lip": [None, None],
        }

        feature_velocity_y = {
            "chin": [None, None],
            "left_eyebrow": [None, None],
            "right_eyebrow": [None, None],
            "nose_bridge": [None, None],
            "nose_tip": [None, None],
            "left_eye": [None, None],
            "right_eye": [None, None],
            "top_lip": [None, None],
            "bottom_lip": [None, None],
        }

        feature_velocity_variance_x = {
            "chin": [None, None],
            "left_eyebrow": [None, None],
            "right_eyebrow": [None, None],
            "nose_bridge": [None, None],
            "nose_tip": [None, None],
            "left_eye": [None, None],
            "right_eye": [None, None],
            "top_lip": [None, None],
            "bottom_lip": [None, None],
        }

        feature_velocity_variance_y = {
            "chin": [None, None],
            "left_eyebrow": [None, None],
            "right_eyebrow": [None, None],
            "nose_bridge": [None, None],
            "nose_tip": [None, None],
            "left_eye": [None, None],
            "right_eye": [None, None],
            "top_lip": [None, None],
            "bottom_lip": [None, None],
        }

        for frame_number in range(2, len(self.frames)):
            for feature in feature_velocity_x:
                distance_1_x, distance_1_y = self.frames[
                    frame_number - 2
                ].getXYDistance(self.frames[frame_number - 1], feature)
                distance_2_x, distance_2_y = self.frames[
                    frame_number - 1
                ].getXYDistance(self.frames[frame_number], feature)
                velocity_x = distance_2_x - distance_1_x
                velocity_y = distance_2_y - distance_1_y
                feature_velocity_x[feature].append(velocity_x)
                feature_velocity_y[feature].append(velocity_y)

        self.chin["velocity_x"], self.chin["velocity_y"] = (
            feature_velocity_x["chin"],
            feature_velocity_y["chin"],
        )
        self.left_eyebrow["velocity_x"], self.left_eyebrow["velocity_y"] = (
            feature_velocity_x["left_eyebrow"],
            feature_velocity_y["left_eyebrow"],
        )
        self.right_eyebrow["velocity_x"], self.right_eyebrow["velocity_y"] = (
            feature_velocity_x["right_eyebrow"],
            feature_velocity_y["right_eyebrow"],
        )
        self.nose_bridge["velocity_x"], self.nose_bridge["velocity_y"] = (
            feature_velocity_x["nose_bridge"],
            feature_velocity_y["nose_bridge"],
        )
        self.nose_tip["velocity_x"], self.nose_tip["velocity_y"] = (
            feature_velocity_x["nose_tip"],
            feature_velocity_y["nose_tip"],
        )
        self.left_eye["velocity_x"], self.left_eye["velocity_y"] = (
            feature_velocity_x["left_eye"],
            feature_velocity_y["left_eye"],
        )
        self.right_eye["velocity_x"], self.right_eye["velocity_y"] = (
            feature_velocity_x["right_eye"],
            feature_velocity_y["right_eye"],
        )
        self.top_lip["velocity_x"], self.top_lip["velocity_y"] = (
            feature_velocity_x["top_lip"],
            feature_velocity_y["top_lip"],
        )
        self.bottom_lip["velocity_x"], self.bottom_lip["velocity_y"] = (
            feature_velocity_x["bottom_lip"],
            feature_velocity_y["bottom_lip"],
        )

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
    def getFrame(self):
        return self.frame

    def getFaceBox(self):
        if self.face_locations is None:
            return None
        return self.face_locations[0]

    def drawFaceLandmarks(self, color=(255, 255, 255), stroke=2):
        # Create blank image of same size as frame
        pil_image = Image.fromarray(self.frame)

        # Create a draw object
        d = ImageDraw.Draw(pil_image)
        d.rectangle([(0, 0), (self.video.width, self.video.height)], fill=(0, 0, 0))

        # Draw face landmarks
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

    def getFeature(self, feature):
        features = {
            "chin": self.chin,
            "left_eyebrow": self.left_eyebrow,
            "right_eyebrow": self.right_eyebrow,
            "nose_bridge": self.nose_bridge,
            "nose_tip": self.nose_tip,
            "left_eye": self.left_eye,
            "right_eye": self.right_eye,
            "top_lip": self.top_lip,
            "bottom_lip": self.bottom_lip,
        }
        return features[feature]

    def getFeatureLength(self, feature):
        if self.getFeature(feature) is None:
            return 0
        return len(self.getFeature(feature))

    def getFeatureCentralPosition(self, feature):
        if self.getFeatureLength(feature) == 0:
            return np.array([0, 0])
        return np.mean(self.getFeature(feature), axis=0)

    def getFeatureDistance(self, other_frame:'Frame', feature):
        """
        Gets the change in central position between a given feature in two different frames

        Parameters
        ----------
        other_frame : Frame
            Frame to compare to
        feature : str
            Feature to compare

        Returns
        ----------
        float
            Distance between the feature centers
        """
        if (
            self.getFeatureLength(feature) == 0
            or other_frame.getFeatureLength(feature) == 0
        ):
            return -1
        return np.linalg.norm(
            self.getFeatureCentralPosition(feature)
            - other_frame.getFeatureCentralPosition(feature)
        )

    def getXYDistance(self, other_frame:'Frame', feature):
        """
        Gets the change in central position between a given feature in two different frames

        Parameters
        ----------
        other_frame : Frame
            Frame to compare to
        feature : str
            Feature to compare

        Returns
        ----------
        float
            Distance between the feature centers
        """
        if (
            self.getFeatureLength(feature) == 0
            or other_frame.getFeatureLength(feature) == 0
        ):
            return -1
        return (
            self.getFeatureCentralPosition(feature)[0]
            - other_frame.getFeatureCentralPosition(feature)[0],
            self.getFeatureCentralPosition(feature)[1]
            - other_frame.getFeatureCentralPosition(feature)[1],
        )
    
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


class Metadata:
    """
    Metadata class

    Attributes
    ----------
    df : pandas.DataFrame
        Metadata

    Public Methods
    ----------
    metadata()
        Get metadata
    """

    def __init__(self, file_path, sample=True):
        """
        Metadata class

        Parameters
        ----------
        file_path : str or dict
            Path to json file or dictionary with paths to json files
        """
        self.sample = sample
        # Load metadata
        if sample:
            self._load_metadata(file_path, multiple=False, sample=sample)
        else:
            self._load_metadata(file_path, multiple=True, sample=sample)

    ## Public methods
    def original(self):
        """
        Get original metadata

        Parameters
        ----------
        None

        Returns
        ----------
        pandas.DataFrame
            Original metadata
        """
        return self.original_df

    def fake(self):
        """
        Get fake metadata

        Parameters
        ----------
        None

        Returns
        ----------
        pandas.DataFrame
            Fake metadata
        """
        return self.fake_df

    def is_sample(self):
        """
        Check if metadata is a sample

        Parameters
        ----------
        None

        Returns
        ----------
        bool
            True if metadata is a sample, False otherwise
        """
        return self.sample

    def get_sample_pairs(self):
        """
        Get sample pairs

        Parameters
        ----------
        None

        Returns
        ----------
        pandas.DataFrame
            Sample pairs
        """
        df = pd.merge(
            self.original_df,
            self.fake_df,
            left_on="id",
            right_on="original",
            how="inner",
            suffixes=("_original", "_fake"),
        ).drop(columns=["original"])
        return df

    ## Private methods
    def _load_metadata(self, file_path, multiple=False, sample=True):
        """
        Load metadata from json file

        Parameters
        ----------
        file_path : str
            Path to json file

        Returns
        ----------
        None
        """
        # Load metadata
        data = None
        if multiple:
            file_paths = file_path
            for folder, file_path in file_paths.items():
                with open(file_path) as json_file:
                    new_data = json.load(json_file)
                new_data = pd.DataFrame.from_dict(data, orient="index")
                new_data.reset_index(inplace=True)
                data["folder"] = folder
                data = pd.concat([data, new_data])
        else:
            with open(file_path) as json_file:
                data = json.load(json_file)
            data = pd.DataFrame.from_dict(data, orient="index")
            data.reset_index(inplace=True)

        # Rename columns
        data.rename(columns={"index": "id"}, inplace=True)
        data.drop(columns=["split"], inplace=True)

        # Remove .mp4 from all ids
        data["id"] = data["id"].apply(lambda x: x[:-4])

        # Split into original and fake datasets
        fake_df = data[data["label"] == "FAKE"]
        original_df = data[data["label"] == "REAL"]

        # Clean datasets
        fake_df.drop(columns=["label"], inplace=True)
        fake_df["original"] = fake_df["original"].apply(lambda x: x[:-4])
        original_df.drop(columns=["label", "original"], inplace=True)

        self.fake_df = fake_df
        self.original_df = original_df
