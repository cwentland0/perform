import unittest
import os

import perform.misc_funcs as misc_funcs


class MiscFuncsTestCase(unittest.TestCase):
    def setUp(self):

        self.test_dir = "test_dir"

        if os.path.isdir(self.test_dir):
            os.rmdir(self.test_dir)

    def tearDown(self):

        if os.path.isdir(self.test_dir):
            os.rmdir(self.test_dir)

    def test_mkdir_shallow(self):

        misc_funcs.mkdir_shallow("", self.test_dir)
        self.assertTrue(os.path.isdir(self.test_dir))


if __name__ == "__main__":
    unittest.main()
