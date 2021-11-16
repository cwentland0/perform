import unittest

from perform.mesh import Mesh


class MeshTestCase(unittest.TestCase):
    def setUp(self):

        self.x_left = 0.0
        self.x_right = 1.0
        self.num_cells = 100
        self.dx = (self.x_right - self.x_left) / self.num_cells

        self.mesh_dict = {}
        self.mesh_dict["x_left"] = self.x_left
        self.mesh_dict["x_right"] = self.x_right
        self.mesh_dict["num_cells"] = self.num_cells

    def test_mesh(self):

        mesh = Mesh(self.mesh_dict)
        self.assertEqual(mesh.x_left, self.x_left)
        self.assertEqual(mesh.x_right, self.x_right)
        self.assertEqual(mesh.num_cells, self.num_cells)
        self.assertEqual(mesh.num_faces, self.num_cells + 1)
        self.assertEqual(mesh.x_face.ndim, 1)
        self.assertEqual(mesh.x_face.shape[0], self.num_cells + 1)
        self.assertEqual(mesh.x_face[0], self.x_left)
        self.assertEqual(mesh.x_face[-1], self.x_right)
        self.assertEqual(mesh.dx, self.dx)
        self.assertEqual(mesh.x_cell.ndim, 1)
        self.assertEqual(mesh.x_cell.shape[0], self.num_cells)
        self.assertEqual(mesh.x_cell[0], self.x_left + self.dx / 2.0)
        self.assertEqual(mesh.x_cell[-1], self.x_right - self.dx / 2.0)


if __name__ == "__main__":
    unittest.main()
