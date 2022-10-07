import argparse
import cv2
import json
import os
import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np

from pytorch3d.io import load_objs_as_meshes, load_obj

import pickle
from tqdm.auto import tqdm
import imageio
import matplotlib.pyplot as plt
import pytorch3d
import pytorch3d.renderer
import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
from pytorch3d.io import load_obj

def load_mesh(path):
    """
    Loads vertices and faces from an obj file.

    Returns:
        vertices (torch.Tensor): The vertices of the mesh (N_v, 3).
        faces (torch.Tensor): The faces of the mesh (N_f, 3).
    """
    vertices, faces, aux = load_obj(path, load_textures=True)
    # faces = faces.verts_idx
    return vertices, faces, aux


def get_device():
    """
    Checks if GPU is available and returns device accordingly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer


def render_mesh(mtl_path, obj_path, output_file, angle, image_size=1024, color=[0.7, 0.7, 1], device=None):
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces, aux = load_mesh(obj_path)
    # vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    # faces = faces.verts_idx.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    # textures = torch.ones_like(vertices)  # (1, N_v, 3)
    # textures = textures * torch.tensor(color)  # (1, N_v, 3)
    # mesh = pytorch3d.structures.Meshes(
    #     verts=vertices,
    #     faces=faces,
    #     textures=pytorch3d.renderer.TexturesVertex(textures),
    # )
    # mesh = mesh.to(device)


    verts_uvs = aux.verts_uvs[None, ...]  # (1, V, 2)
    faces_uvs = faces.textures_idx[None, ...]  # (1, F, 3)
    tex_maps = aux.texture_images

    # tex_maps is a dictionary of {material name: texture image}.
    # Take the first image:
    texture_image = list(tex_maps.values())[0]
    texture_image = texture_image[None, ...]  # (1, H, W, 3)

    # Create a textures object
    tex = pytorch3d.renderer.Textures(verts_uvs=verts_uvs, faces_uvs=faces_uvs, maps=texture_image)

    vertices[:,0]+=100
    vertices[:,1]-=500
    print(vertices.shape)

    # apply a translation

    # Initialise the mesh with textures
    mesh = pytorch3d.structures.Meshes(verts=[vertices], faces=[faces.verts_idx], textures=tex)

    mesh = mesh.to(device)

    # camera_position = pytorch3d.renderer.cameras.camera_position_from_spherical_angles(distance=2,azimuth=0, elevation=angle, degrees=True)

    R, T = pytorch3d.renderer.cameras.look_at_view_transform(dist = 2000, elev=0, azim=angle)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(device=device, R=R, T=T)

    # cam = pytorch3d.renderer.cameras.CamerasBase()
    # print(cam.get_camera_center())

    rend = renderer(mesh, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)

    plt.imsave(output_file, rend)


if __name__ == "__main__":
   
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_file", type=str, required=True, help="The path to the obj file.")
    parser.add_argument("--mtl_file", type=str, required=True, help="The path to the mtl file.")
    parser.add_argument("--subject_name", type=str, required=False, help="The name of the subject to extract information for")
    parser.add_argument("--number_intervals", type=int, required=False, default=10, help="Number of images to save, even intervals in 360 degrees")
    args = parser.parse_args()

    obj_filename = "/home/dupmaka/facescape_old/toolkit/bp4d_output_neutral/F008.obj"

    angles = np.linspace(0,360, args.number_intervals)
    print(args.obj_file[-8:-4])
    for angle in angles:
        render_mesh(args.mtl_file, args.obj_file, "multiview_render/"+args.obj_file[-8:-4]+"_"+ str(angle)+ "_.jpeg", angle, image_size=256, color=[0.7, 0.7, 1], device=None)
