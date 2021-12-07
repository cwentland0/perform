import unittest
import os

from constants import TEST_DIR, del_test_dir
import perform.misc_funcs as misc_funcs


class MiscFuncsTestCase(unittest.TestCase):
    def setUp(self):

        del_test_dir()

    def tearDown(self):

        del_test_dir()

    def test_mkdir_shallow(self):

        misc_funcs.mkdir_shallow("", TEST_DIR)
        self.assertTrue(os.path.isdir(TEST_DIR))


if __name__ == "__main__":
    unittest.main()
