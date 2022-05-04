import argparse
import cv2
import json
import os
from multiprocessing import Pool


def parse_input(path):
    with open(path, 'r') as f:
        return json.load(f)

def create_output_dir(output_root, video_paths):
    output_dirs = []
    for video_path in video_paths:
        video_name = os.path.basename(video_path).split('.')[0]
        output_dir = os.path.join(output_root, video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_dirs.append(output_dir)
    return output_dirs

def extract_from_video(video_path, output_dir, input_file):
    failed = []
    cap = cv2.VideoCapture(video_path)
    for frame_num in input_file[video_path]:
        cap.set(cv2.CAP_PROP_FRAME_COUNT, frame_num - 1)
        res, frame = cap.read()
        if res:
            path = os.path.join(output_dir, f"{frame_num}.jpg")
            cv2.imwrite(path, frame)
        else:
            print(f"Failed to extract {video_path} frame: {frame_num}")
            failed.append(frame_num)
    return (video_path, failed)


def main(args):
    input_file = parse_input(args.input)
    video_paths = list(input_file.keys())
    output_dirs = create_output_dir(args.output_root, video_paths)

    failed = None
    tasks = [(video_path, output_dir, input_file) for video_path, output_dir in zip(video_paths, output_dirs)]
    with Pool(args.workers) as p:
        failed = p.starmap(extract_from_video, tasks)
        failed = {video_path: frames for video_path, frames in failed}
        for video_path in list(failed.keys()):
            if len(failed[video_path]) == 0:
                failed.pop(video_path)
        
    if failed:
        failed_path = os.path.join(args.output_root, "failed.json")
        with open(failed_path, "w+") as f:
            json.dump(failed, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="The path to the input file.")
    parser.add_argument("--output_root", type=str, required=True, help="The path to the output directory.")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers to use.")
    args = parser.parse_args()
    main(args)
