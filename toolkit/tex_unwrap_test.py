import cv2
from src.facescape_fitter import facescape_fitter
import numpy as np
from src.facescape_bm import facescape_bm
from src.renderer import render_cvcam
import timeit
# import cupy as cp
import csv
import numpy as np, cv2, trimesh
from src.facescape_fitter import facescape_fitter
from src.renderer import render_orthcam
from src.renderer import render_cvcam


np.random.seed(1000)

# Initialize model and fitter
fs_fitter = facescape_fitter(fs_file="facescape_bm_v1.6_847_50_52_id_front.npz", kp2d_backend='dlib')  # or 'face_alignment'

# Fit id to image
src_path = "./test_data/0.jpg"
src_img = cv2.imread(src_path)
kp2d = fs_fitter.detect_kp2d(src_img)  # extract 2D key points
mesh, params, mesh_verts_img = fs_fitter.fit_kp2d(kp2d)  # fit model
id, _, scale, trans, rot_vector = params


# Get texture
# add extra face 
texture = fs_fitter.get_texture(src_img, mesh_verts_img, mesh)
filename = './demo_output/test_mesh.jpg'
cv2.imwrite(filename, texture)

# Save base mesh
mesh.export(output_dir='./demo_output', file_name='test_mesh', texture_name='test_mesh.jpg', enable_vc=False, enable_vt=False)

##############################################################
# NEW CODE START
##############################################################
from pymeshfix._meshfix import PyTMesh
from pymeshfix.examples import planar_mesh
import pymeshfix as mf

import pytorch3d
from pytorch3d.io import load_obj

# faces = self.fv_indices


# print(type(mesh_verts), mesh_verts.shape)
# print(type(faces), faces.shape)
# print("created meshfix")


# holes = meshfix.extract_holes()
path = "demo_output/test_mesh.obj"
vertices, faces, _ = load_obj(path)
faces = faces.verts_idx

# print(mesh.vertices)
# faces = []
# for f in mesh.faces:
#     faces.append(list(f[0]))
# faces = np.array(faces)

meshfix = mf.MeshFix(vertices.numpy(), faces.numpy())
meshfix.repair(remove_smallest_components=False)

# vertices, faces = mf.clean_from_arrays(np.array(mesh.vertices), faces)

print("repaired")

vertices = meshfix.points()
faces = meshfix.faces()

mesh.vertices = vertices
# mesh.fv_indices = faces

mesh.create(vertices=mesh.vertices,
            faces_v=faces,   # face vertices
            faces_vt=fs_fitter.ft_indices,  # face texture coordinates
            texcoords=fs_fitter.texcoords   # uv coordinates
    )


# (id, exp, scale, trans, rot_vector) = params 
mesh_verts_img = vertices #facescape_fitter.project(vertices, rot_vector, scale, trans)

##############################################################
# NEW CODE END
##############################################################

# Get texture
texture = fs_fitter.get_texture(src_img, mesh_verts_img, mesh)
filename = './demo_output/test_mesh1.jpg'
cv2.imwrite(filename, texture)

# Save base mesh
mesh.export(output_dir='./demo_output', file_name='test_mesh1', texture_name='test_mesh1.jpg', enable_vc=False, enable_vt=False)
