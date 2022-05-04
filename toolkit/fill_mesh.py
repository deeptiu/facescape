from pytorch3d.io import load_obj
import numpy as np
import scipy

def add_new_faces(faces_idx, faces_tex, coordinates, vertices):
    fv_list = []
    ft_list = []
    indices = []
    for c in coordinates:
        dist = scipy.spatial.distance.cdist(vertices, [c])
        min_idx = np.argmin(dist) + 1
        fv_list.append(min_idx)

        for i in range(len(faces_idx)):
            if faces_idx[i][0] == min_idx:
                ft_list.append(faces_tex[i][0])
                break
            elif faces_idx[i][1] == min_idx:
                ft_list.append(faces_tex[i][1])
                break
            elif faces_idx[i][2] == min_idx:
                ft_list.append(faces_tex[i][2])
                break

    fv_list = np.array(fv_list)
    ft_list = np.array(ft_list)

    for i in range(len(fv_list)-2):
        faces_idx = np.append(faces_idx, [[fv_list[i], fv_list[i+1], fv_list[i+2]]], axis=0)
        faces_tex = np.append(faces_tex, [[ft_list[i], ft_list[i+1], ft_list[i+2]]], axis=0)

    return faces_idx, faces_tex

def load_file(path):
    # path = "demo_output/original_mesh.obj"
    vertices, faces, _ = load_obj(path)
    faces_idx = faces.verts_idx
    faces_tex = faces.textures_idx

    vertices = vertices.numpy()
    faces_idx = faces_idx.numpy()
    faces_tex = faces_tex.numpy()

    return vertices, faces_idx, faces_tex

def add_picked_points(vertices, faces_idx, faces_tex):
    with open('picked_point.txt') as f:
        lines = f.readlines()

    coordinates_left_eye = []
    coordinates_right_eye = []
    coordinates_mouth = []
    for l in lines:
        l_split = l.split()
        part = l_split[0]
        x = float(l_split[1])
        y = float(l_split[2])
        z = float(l_split[3])
        if part == "left_eye":
            coordinates_left_eye.append([x, y, z])
        elif part == "right_eye":
            coordinates_right_eye.append([x, y, z])
        elif part == "mouth":
            coordinates_mouth.append([x, y, z])

    coordinates_left_eye = np.array(coordinates_left_eye)
    coordinates_right_eye = np.array(coordinates_right_eye)
    coordinates_mouth = np.array(coordinates_mouth)

    faces_idx, faces_tex = add_new_faces(faces_idx, faces_tex, coordinates_left_eye, vertices)
    faces_idx, faces_tex = add_new_faces(faces_idx, faces_tex, coordinates_right_eye, vertices)
    faces_idx, faces_tex = add_new_faces(faces_idx, faces_tex, coordinates_mouth, vertices)

    return faces_idx, faces_tex







    



