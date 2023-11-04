#!/usr/bin/env python

# Copyright 2023, Gavin E. Crooks and contributors
#
# This source code is licensed under the MIT License
# found in the LICENSE file in the root directory of this source tree.


import json
import os
from pathlib import Path
import math
from itertools import combinations_with_replacement, permutations, product
from math import pow, sqrt

import numpy as np
from scipy.spatial import ConvexHull
from stl import mesh  # numpy-stl

target_volume = 38**3  # So that cube has faces 38mm on a side.

familes = (
    "platonic",
    "archimedean",
    "catalan",
    "johnson",
    "prism_antiprism",
    "pyramid_dipyramid",
)


def triangulate(vertices):
    # Assumes convex shape, so wouldn't work with stellated solids.
    hull = ConvexHull(vertices)
    indices = hull.simplices

    center = np.mean(vertices, axis=0)

    new_indices = []

    for i, j, k in indices:
        v0 = vertices[i]
        v1 = vertices[j]
        v2 = vertices[k]

        normal = np.cross(v1 - v0, v2 - v0)

        d = np.dot(-v0, normal)

        if d < 0:
            new_indices.append([i, j, k])
        else:
            new_indices.append([k, j, i])

    return np.asarray(new_indices)


def save_stl(vertices, faces, name):
    polyhedron = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            polyhedron.vectors[i][j] = vertices[f[j], :]

    volume, cog, inertia = polyhedron.get_mass_properties()

    scale = (target_volume / volume) ** (1 / 3)
    polyhedron.x *= scale
    polyhedron.y *= scale
    polyhedron.z *= scale
    volume, cog, inertia = polyhedron.get_mass_properties()

    Path("shapes").mkdir(parents=True, exist_ok=True)
    polyhedron.save(os.path.join('shapes', name + ".stl"))


def normalize_name(name, prefix=""):
    return prefix + name.lower().replace(" ", "_")


if __name__ == "__main__":
    for fam in familes:
        with open(os.path.join("vertices", fam + ".json")) as f:
            file_contents = f.read()
            parsed_json = json.loads(file_contents)

        for shape in parsed_json:
            vertices = np.asarray(parsed_json[shape]["vertices"])
            indices = triangulate(vertices)
            save_stl(vertices, indices, normalize_name(shape, fam + "_"))
