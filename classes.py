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
import os

class Video:
    def __init__(self, metadata:'Metadata', id, frameLimit=150, sample=True, tq="notebook"):
        self.face_detections = []  # Initialize the list to track face detections
        self.metadata = metadata
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
        feature_velocity_var_x, feature_velocity_var_y = self._get_all_velocity_variance()
        return feature_velocity_var_x[feature], feature_velocity_var_y[feature]
    
    def getFaceDetectionPercentage(self):
        """
        Compute the percentage of successful face detections throughout the video.

        Returns
        ----------
        float
            Percentage of successful face detections.
        """
        successful_detections = sum(self.face_detections)
        total_frames = len(self.face_detections)
        return (successful_detections / total_frames) * 100
    
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
            else c.FULL_DATA_DIR + '/' + folder_prefix + metadata.getFolder(id)
        )
        file_path = folder_path
        file_name = "/" + str(id) + ".mp4"
        if sample:
            file_path += file_name
        else:
            file_path += file_name
        print(file_path)
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
        # self.duration = self.frame_count / self.fps
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        pass

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
        for frame in self.frames:
            self.face_detections.append(frame.hasFace())
    
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
            if not frame.hasFace():
                print("Frame has no face")
                for feature in features:
                    features[feature].append(None)
                continue
            for feature in features:
                f = frame.getFeature(feature)
                if f is None:
                    print(f"Feature {feature} is None")
                features[feature].append(f)

        self.chin = pd.Series(features["chin"]) if features["chin"] is not None else None
        self.left_eyebrow = pd.Series(features["left_eyebrow"]) if features["left_eyebrow"] is not None else None
        self.right_eyebrow = pd.Series(features["right_eyebrow"]) if features["right_eyebrow"] is not None else None
        self.nose_bridge = pd.Series(features["nose_bridge"]) if features["nose_bridge"] is not None else None
        self.nose_tip = pd.Series(features["nose_tip"]) if features["nose_tip"] is not None else None
        self.left_eye = pd.Series(features["left_eye"]) if features["left_eye"] is not None else None
        self.right_eye = pd.Series(features["right_eye"]) if features["right_eye"] is not None else None
        self.top_lip = pd.Series(features["top_lip"]) if features["top_lip"] is not None else None
        self.bottom_lip = pd.Series(features["bottom_lip"]) if features["bottom_lip"] is not None else None

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
            if not frame.hasFace():
                for feature in feature_centers:
                    feature_centers[feature].append(None)
                continue
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
            if not self.frames[frame_number].hasFace() or not self.frames[
                frame_number - 1].hasFace() or not self.frames[frame_number - 2].hasFace():
                for feature in feature_distances:
                    feature_distances[feature].append(None)
                continue
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

        for frame_number in range(2, len(self.frames)):
            if not self.frames[frame_number].hasFace() or not self.frames[
                frame_number - 1].hasFace() or not self.frames[frame_number - 2].hasFace():
                for feature in feature_velocity_x:
                    feature_velocity_x[feature].append(None)
                    feature_velocity_y[feature].append(None)
                continue
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

    def _get_all_velocity_variance(self):
        feature_names = [
            "chin",
            "left_eyebrow",
            "right_eyebrow",
            "nose_bridge",
            "nose_tip",
            "left_eye",
            "right_eye",
            "top_lip",
            "bottom_lip"
        ]
        feature_velocity_var_x = {}
        feature_velocity_var_y = {}
        for feature in feature_names:
            velocity_x = self.getFeatureDf(feature)["velocity_x"]
            velocity_y = self.getFeatureDf(feature)["velocity_y"]
            velocity_x = pd.Series([v for v in velocity_x if v is not None])
            velocity_y = pd.Series([v for v in velocity_y if v is not None])
            feature_velocity_var_x[feature] = np.var(velocity_x)
            feature_velocity_var_y[feature] = np.var(velocity_y)
        return feature_velocity_var_x, feature_velocity_var_y

    def _export_features(self, folder_path):
        """
        Compile and export all features to json

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        compiled_frames = {}
        for frame in self.frames:
            frame_number = frame.frame_number
            landmarks = frame._get_all_landmarks_as_dict()
            frame_data = {}

            # Landmarks
            for feature in landmarks:
                if landmarks[feature] is None:
                    frame_data[feature] = None
                else:
                    frame_data[feature] = landmarks[feature]
            
            # Distances
            for feature in landmarks:
                if landmarks[feature] is None or frame_number < 1:
                    frame_data[f"{feature}_xdist"] = None
                    frame_data[f"{feature}_ydist"] = None
                else:
                    if compiled_frames[frame_number - 1][f"{feature}_xdist"] is None or compiled_frames[frame_number - 1][f"{feature}_ydist"] is None:
                        frame_data[f"{feature}_xdist"] = None
                        frame_data[f"{feature}_ydist"] = None
                        continue
                    xdist, ydist = frame.getXYDistance(self.frames[frame_number - 1], feature)
                    frame_data[f"{feature}_xdist"] = xdist
                    frame_data[f"{feature}_ydist"] = ydist

            # Velocities
            for feature in landmarks:
                if landmarks[feature] is None or frame_number < 2:
                    frame_data[f"{feature}_xvel"] = None
                    frame_data[f"{feature}_yvel"] = None
                else:
                    distance_1_x = compiled_frames[frame_number - 2][f"{feature}_xdist"]
                    distance_1_y = compiled_frames[frame_number - 2][f"{feature}_ydist"]
                    distance_2_x = compiled_frames[frame_number - 1][f"{feature}_xdist"]
                    distance_2_y = compiled_frames[frame_number - 1][f"{feature}_ydist"]
                    if distance_1_x is None or distance_1_y is None or distance_2_x is None or distance_2_y is None:
                        frame_data[f"{feature}_xvel"] = None
                        frame_data[f"{feature}_yvel"] = None
                        continue
                    xvel = distance_2_x - distance_1_x
                    yvel = distance_2_y - distance_1_y
                    frame_data[f"{feature}_xvel"] = xvel
                    frame_data[f"{feature}_yvel"] = yvel

            compiled_frames[frame_number] = frame_data
        feature_names = [
            "chin",
            "left_eyebrow",
            "right_eyebrow",
            "nose_bridge",
            "nose_tip",
            "left_eye",
            "right_eye",
            "top_lip",
            "bottom_lip"
        ]
        overall_features = {}
        for feature in feature_names:
            feature_velocity_var_x, feature_velocity_var_y = self.getVelocityVariance(feature)
            overall_features[feature + "_xvel_var"] = feature_velocity_var_x
            overall_features[feature + "_yvel_var"] = feature_velocity_var_y
        overall_features["face_detection_percentage"] = self.getFaceDetectionPercentage()
        overall_features["label"] = self.metadata.label(self.id)
        compiled_frames["overall_features"] = overall_features

        with open(folder_path + self.id + ".json", "w") as outfile:
            json.dump(compiled_frames, outfile)


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
    def frame_number(self):
        return self.frame_number
    
    def getFrame(self):
        return self.frame

    def getFaceBox(self):
        if self.face_locations is None:
            return None
        return self.face_locations[0]
    
    def hasFace(self):
        """
        Check if a face was detected in the frame.

        Returns
        ----------
        bool
            True if a face was detected, False otherwise.
        """
        # Check if all of the facial landmarks are not empty
        landmark_attributes = [
            self.chin, self.left_eyebrow, self.right_eyebrow,
            self.nose_bridge, self.nose_tip, self.left_eye,
            self.right_eye, self.top_lip, self.bottom_lip
        ]
        
        for landmark in landmark_attributes:
            if landmark is None:
                return False
        return True
    
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
        if (not self.hasFace() or not other_frame.hasFace()):
            return None
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
        if (not self.hasFace() or not other_frame.hasFace()):
            return None
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
    
    def _get_all_landmarks_as_dict(self):
        """
        Get all landmarks as a dictionary

        Parameters
        ----------
        None

        Returns
        ----------
        dict
            Dictionary of all landmarks
        """
        return {
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
    
    def label(self, id):
        """
        Get label of video

        Parameters
        ----------
        id : str
            Video id

        Returns
        ----------
        str
            Label of video
        """
        if id in self.original_df["id"].values:
            return "REAL"
        elif id in self.fake_df["id"].values:
            return "FAKE"
        else:
            return None

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
    
    def balanced(self):
        # Balance the dataframe so that original and fake videos alternate in the dataframe
        original_df = self.original()
        fake_df = self.fake()
        balanced_df = []
        for i in range(len(original_df)):
            balanced_df.append(original_df.iloc[i])
            balanced_df.append(fake_df.iloc[i])
        balanced_df = pd.DataFrame(balanced_df)
        return balanced_df
    
    def df(self):
        """
        Get metadata

        Parameters
        ----------
        None

        Returns
        ----------
        pandas.DataFrame
            Metadata
        """
        return self.df
    
    def getFolder(self, id):
        """
        Get folder of video

        Parameters
        ----------
        id : str
            Video id

        Returns
        ----------
        str
            Folder of video
        """
        return self.df[self.df["id"] == id]["folder"].values[0]

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
            for folder in os.listdir(file_path):
                folder_number = folder.split("_")[-1]
                with open(file_path + '/' + folder + "/metadata.json") as json_file:
                    new_data = json.load(json_file)
                new_data = pd.DataFrame.from_dict(new_data, orient="index")
                new_data.reset_index(inplace=True)
                new_data["folder"] = folder_number
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
        self.df = data[['id', 'label', 'folder']]

        # Split into original and fake datasets
        fake_df = data[data["label"] == "FAKE"]
        original_df = data[data["label"] == "REAL"]

        # Clean datasets
        fake_df.drop(columns=["label"], inplace=True)
        fake_df["original"] = fake_df["original"].apply(lambda x: x[:-4])
        original_df.drop(columns=["label", "original"], inplace=True)

        self.fake_df = fake_df
        self.original_df = original_df
