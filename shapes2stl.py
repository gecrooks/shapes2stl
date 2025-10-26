#!/usr/bin/env -S uv run --script

# Copyright 2023, Gavin E. Crooks and contributors
#
# This source code is licensed under the MIT License
# found in the LICENSE file in the root directory of this source tree.


import json
import os
from pathlib import Path
import math

import numpy as np
import trimesh

target_volume = 38**3  # So that cube has faces 38mm on a side.

familes = (
    "platonic",
    "archimedean",
    "catalan",
    "johnson",
    "prism_antiprism",
    "pyramid_dipyramid",
)


def convex_mesh(vertices):
    """Return the convex hull of the supplied vertices as a Trimesh."""
    polygon = trimesh.convex.convex_hull(vertices)
    polygon.fix_normals()

    volume = polygon.volume
    scale = (target_volume / volume) ** (1 / 3)
    polygon.apply_scale(scale)

    return polygon


def save_stl(polyhedron, name):
    output_dir = Path("shapes")
    output_dir.mkdir(parents=True, exist_ok=True)
    polyhedron.export(output_dir / f"{name}.stl")


def normalize_name(name, prefix=""):
    return prefix + name.lower().replace(" ", "_")


if __name__ == "__main__":
    for fam in familes:
        with open(os.path.join("vertices", fam + ".json")) as f:
            file_contents = f.read()
            parsed_json = json.loads(file_contents)

        for shape in parsed_json:
            vertices = np.asarray(parsed_json[shape]["vertices"])
            mesh = convex_mesh(vertices)
            save_stl(mesh, normalize_name(shape, fam + "_"))
