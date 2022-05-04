import argparse
import cv2
import json
import os
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np

# based on keypoints only get frames that have open eyes
def check_eyes_open(keypoints):
    
    # left eye keypoints 37, 38, 39, 40, 41, 42
    # right eye keypoints 43, 44, 45, 46, 47, 48

    # EAR (eye aspect ratio)
    # subtract 1 since keypoint indexing begins at 1 on the diagram
    left_ear = (np.linalg.norm(keypoints[38-1], keypoints[42-1]) + np.linalg.norm(keypoints[39-1], keypoints[41-1]))/(2*np.linalg.norm(keypoints[37-1], keypoints[40-1]))
    right_ear = (np.linalg.norm(keypoints[44-1], keypoints[48-1]) + np.linalg.norm(keypoints[45-1], keypoints[47-1]))/(2*np.linalg.norm(keypoints[43-1], keypoints[46-1]))

    if left_ear < 0.3 or right_ear < 0.3:
        return False
    return True

def check_gaze_forward(keypoints, src_path):

    
    return True
def get_all_unique_subjects(base_path):

    all_subjects = []
    video_path = os.path.join(base_path, "2D_Video")

    all_files = os.listdir(video_path)
    for f in all_files:
        subject_name = f[:-4].split("_")[0]
        if subject_name not in all_subjects:
            all_subjects.append(subject_name)

    return all_subjects

def all_frames_per_subject(subject_name, base_path):

    facs_path = os.path.join(base_path, "FACS/OCC")
    video_path = os.path.join(base_path, "2D_Video")

    facs_files = [f for f in listdir(facs_path) if isfile(join(facs_path, f))]
    # video_files = [f for f in listdir(video_path) if isfile(join(video_path, f))]

    all_frames = []
    video_filenames = []

    for i,f in enumerate(facs_files):
        file_split = f.split("_")
        subject = file_split[0]
        if subject == subject_name:
            sequence_num = file_split[1]
            video_filename = video_path+"/"+subject_name+"_"+sequence_num+".avi"
            video_filenames.append(video_filename)
            df = pd.read_csv(facs_path+"/"+f)
            df["video_sequence_num"] = np.ones(df.shape[0])*int(sequence_num)
            df = df.rename({"1": "frame_id", "1.1": "1"}, axis='columns')
            all_frames.append(df)

    if len(all_frames) < 1:
        return None, None
    result = pd.concat(all_frames)

    return result, video_filenames

# find first frame with the max au occurances
def find_max(subject_name, all_frames, base_path, au):

    frame = None
    return frame

# find first frame with the max 0's 
def find_neutral_frame(subject_name, base_path, all_frames, all_video_paths):

    video_paths = []
    frame_ids = []
    # find frame with most 0's in row 
    all_aus = all_frames.loc[:, all_frames.columns != "frame_id"]
    all_aus = all_aus.loc[:, all_aus.columns != "video_sequence_num"]
    sums = all_aus.sum(axis=1)

    np_sums = sums.to_numpy()
    min_frame_ids = np.argwhere(np_sums == 0)
    min_frame_id = sums.idxmin()

    # frame must satisfy two conditions
    # 1) eyes must be open
    # 2) head turned to front
    for frame_id in min_frame_ids:

        neutral_row = all_frames.iloc[frame_id]
        sequence_id = neutral_row["video_sequence_num"]
        frame_id = int(neutral_row["frame_id"])
        video_path = os.path.join(base_path, "2D_Video") + "/"+ subject_name+"_"+str(int(sequence_id))+".avi"
        frame_ids.append(frame_id)
        video_paths.append(video_path)

    return video_paths, frame_ids

def extract_from_video(video_paths, output_dir, frame_ids, subject_name):
    # print("frame ids", frame_ids)
    failed = []
    paths = []
    failed_video_path = []
    for video_path,frame_num in zip(video_paths, frame_ids):
        cap = cv2.VideoCapture(video_path)
        print("cap", cv2.CAP_PROP_FRAME_COUNT)
        cap.set(cv2.CAP_PROP_FRAME_COUNT, frame_num - 1)
        res, frame = cap.read()
        if res:
            path = os.path.join(output_dir, f"{subject_name}_{frame_num}.jpg")
            cv2.imwrite(path, frame)
            paths.append(path)
        else:
            print(f"Failed to extract {video_path} frame: {frame_num}")
            failed.append(frame_num)
            failed_video_path.append(video_path)
    return (failed_video_path, failed), paths

if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=2 python assign_au.py --base_path /data/datasets/EB+ --subject_name M010

    base_path = "/data/datasets/EB+"
    subject_name = "M010"
    all_frames_per_subject(subject_name, base_path)

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True, help="The path to the base directory containing FACS and 2D_Video.")
    parser.add_argument("--subject_name", type=str, required=False, help="The name of the subject to extract information for")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers to use.")
    args = parser.parse_args()

    get_all_unique_subjects(args.base_path)

    # per subject
    all_frames, all_video_paths = all_frames_per_subject(args.subject_name, args.base_path)

    video_path, frame_ids = find_neutral_frame(args.subject_name, args.base_path, all_frames, all_video_paths)

    output_dir = "all_neutral_frames/"
    extract_from_video(video_path, output_dir, frame_ids, args.subject_name)