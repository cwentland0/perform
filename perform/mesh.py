import numpy as np

from perform.constants import REAL_TYPE


class Mesh:
    """Class for defining computational mesh.

    Currently, this directly implements a uniform mesh.
    If, in the future, non-uniform and/or adaptive meshes are implemented, this should serve
    as the base class for those more specific child classes.

    Args:
        mesh_dict:

    Attributes:
        x_left: Coordinate of left-most finite volume face, indicating the inlet boundary face.
        x_right: Coordinate of right-most finite volume face, indicating the outlet boundary face.
        num_cells: Number of finite volume cells, not including boundary ghost cells.
        num_faces: Number of finite volume faces, including boundary faces.
        x_face: NumPy array of coordinates of finite volume cell faces.
        x_cell: NumPy array of coordinates of finite volume cell centers.
        dx: Fixed finite "volume" cell length, in meters.
    """

    def __init__(self, mesh_dict):

        self.x_left = float(mesh_dict["x_left"])
        self.x_right = float(mesh_dict["x_right"])
        self.num_cells = int(mesh_dict["num_cells"])
        self.num_faces = self.num_cells + 1
        self.x_face = np.linspace(self.x_left, self.x_right, self.num_cells + 1, dtype=REAL_TYPE)
        self.x_cell = (self.x_face[1:] + self.x_face[:-1]) / 2.0
        self.dx = self.x_face[1] - self.x_face[0]
