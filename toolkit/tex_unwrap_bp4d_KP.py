import cv2
from src.facescape_fitter import facescape_fitter
import numpy as np
from src.facescape_bm import facescape_bm
from src.renderer import render_cvcam
import timeit
# import cupy as cp
import csv
import numpy as np, trimesh
from src.facescape_fitter import facescape_fitter
from src.renderer import render_orthcam
from src.renderer import render_cvcam
from src.mesh_obj import mesh_obj
from fill_mesh import *
import os
from assign_au import *

np.random.seed(1000)

# Initialize model and fitter
fs_fitter = facescape_fitter(fs_file="facescape_bm_v1.6_847_50_52_id_front.npz", kp2d_backend='dlib')  # or 'face_alignment'

# inital parameters 
id = fs_fitter.id_mean
exp = np.array([1] + [0] * 51, dtype=np.float32)
rot_vector = np.array([0, 0, 0], dtype=np.double)
trans = np.array([0, 0])
scale = 1.

mesh_verts = fs_fitter.shape_bm_core.dot(id).dot(exp).reshape((-1, 3))
vertices = fs_fitter.project(mesh_verts, rot_vector, scale, trans,  keepz=True)

# add new faces to account for open cavities
fs_fitter.fv_indices, fs_fitter.ft_indices = add_picked_points(vertices, fs_fitter.fv_indices, fs_fitter.ft_indices)

# save new base model (additional step, not necessary for functionality)
mesh, params, mesh_verts_img = fs_fitter.save_dm_mesh()
mesh.export(output_dir='./bp4d_output', file_name='original_mesh', texture_name=None, enable_vc=False, enable_vt=True)

# PROCESS FIRST FRAME FROM EACH VIDEO
base_path = "/data/datasets/EB+"
path_vids = "/data/datasets/EB+/2D_Video"
dir_list = os.listdir(path_vids)
output_dir = "neutral_frames/"

all_subjects = get_all_unique_subjects(base_path)
all_subjects = all_subjects[:2]

for subject_name in all_subjects:
    print(subject_name)

    all_frames, all_video_paths = all_frames_per_subject(subject_name, base_path)

    video_path, frame_ids = find_neutral_frame(subject_name, base_path, all_frames, all_video_paths)

    tup, paths = extract_from_video(video_path, output_dir, frame_ids, subject_name)

    if len(paths) < 1:
        continue
    # Fit id to image
    src_path = paths[0]
    src_img = cv2.imread(src_path)
    kp2d = fs_fitter.detect_kp2d(src_img)  # extract 2D key points
    mesh, params, mesh_verts_img = fs_fitter.fit_kp2d(kp2d)  # fit model

    # Get texture
    try:
        texture = fs_fitter.get_texture(src_img, mesh_verts_img, mesh)
        filename = f'./bp4d_output_neutral/{subject_name}.jpg'
        cv2.imwrite(filename, texture)

        # Save base mesh
        mesh.export(output_dir='./bp4d_output_neutral', file_name=f'{subject_name}', texture_name=f'{subject_name}.jpg', enable_vc=False, enable_vt=True)
    except:
        print("Encountered some cv error")

# for f in dir_list:
#     print(f)
#     full_path = path_vids + "/" + f
#     vidcap = cv2.VideoCapture(full_path)
#     success,image = vidcap.read()
#     # image = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))
#     filename = f"bp4d_frames/{f}.jpg"
#     cv2.imwrite(filename, image)     # save frame as JPEG file   

#     # PER IMAGE PROCESSING 

#     # Fit id to image
#     src_path = "./" + filename
#     print(src_path)
#     src_img = cv2.imread(src_path)
#     kp2d = fs_fitter.detect_kp2d(src_img)  # extract 2D key points
#     mesh, params, mesh_verts_img = fs_fitter.fit_kp2d(kp2d)  # fit model

#     # Get texture
#     try:
#         texture = fs_fitter.get_texture(src_img, mesh_verts_img, mesh)
#         filename = f'./bp4d_output/{f}.jpg'
#         cv2.imwrite(filename, texture)

#         # Save base mesh
#         mesh.export(output_dir='./bp4d_output', file_name=f'{f}', texture_name=f'{f}.jpg', enable_vc=False, enable_vt=True)
#     except:
#         print("Encountered some cv error")