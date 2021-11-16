import unittest
import os

import numpy as np

import perform.input_funcs as input_funcs


class InputParsersTestCase(unittest.TestCase):
    def setUp(self):

        self.test_int = 1
        self.test_float = 1.0
        self.test_bool = False
        self.test_str = "test"
        self.test_list = [0.0, 1.0, 2.0]
        self.test_lol = [[0, 1], [2, 3]]

        self.test_dict = {}
        self.test_dict["test_int"] = self.test_int
        self.test_dict["test_float"] = self.test_float
        self.test_dict["test_bool"] = self.test_bool
        self.test_dict["test_str"] = self.test_str
        self.test_dict["test_list"] = self.test_list
        self.test_dict["test_lol"] = self.test_lol

        # generate test input file
        with open("test_input_funcs.inp", "w") as f:
            f.write("test_int = " + str(self.test_int) + "\n")
            f.write("test_float = " + str(self.test_float) + "\n")
            f.write("test_bool = " + str(self.test_bool) + "\n")
            f.write('test_str = "' + self.test_str + '"\n')
            f.write("test_list = " + str(self.test_list) + "\n")
            f.write("test_lol = " + str(self.test_lol) + "\n")

    def tearDown(self):

        os.remove("test_input_funcs.inp")

    def test_catch_input(self):

        # test exact return
        out_int = input_funcs.catch_input(self.test_dict, "test_int", None)
        out_float = input_funcs.catch_input(self.test_dict, "test_float", None)
        out_bool = input_funcs.catch_input(self.test_dict, "test_bool", None)
        out_str = input_funcs.catch_input(self.test_dict, "test_str", None)
        self.assertEqual(out_int, self.test_int)
        self.assertEqual(out_float, self.test_float)
        self.assertEqual(out_bool, self.test_bool)
        self.assertEqual(out_str, self.test_str)

        # test casting by defaults
        out_int = input_funcs.catch_input(self.test_dict, "test_int", -1)
        out_float = input_funcs.catch_input(self.test_dict, "test_float", -1.0)
        out_bool = input_funcs.catch_input(self.test_dict, "test_bool", True)
        out_str = input_funcs.catch_input(self.test_dict, "test_str", "test2")
        self.assertEqual(out_int, self.test_int)
        self.assertEqual(out_float, self.test_float)
        self.assertEqual(out_bool, self.test_bool)
        self.assertEqual(out_str, self.test_str)

        # test default assignment
        out_int = input_funcs.catch_input(self.test_dict, "no_int", self.test_int)
        out_float = input_funcs.catch_input(self.test_dict, "no_float", self.test_float)
        out_bool = input_funcs.catch_input(self.test_dict, "no_bool", self.test_bool)
        out_str = input_funcs.catch_input(self.test_dict, "no_str", self.test_str)
        self.assertEqual(out_int, self.test_int)
        self.assertEqual(out_float, self.test_float)
        self.assertEqual(out_bool, self.test_bool)
        self.assertEqual(out_str, self.test_str)

        # test casting int to float
        out_float = input_funcs.catch_input(self.test_dict, "test_int", -1.0)
        self.assertEqual(out_float, self.test_float)

    def test_catch_list(self):

        # check exact return
        out_list = input_funcs.catch_list(self.test_dict, "test_list", [None])
        out_lol = input_funcs.catch_list(self.test_dict, "test_lol", [[None]], len_highest=len(self.test_lol))
        self.assertEqual(out_list, self.test_list)
        self.assertEqual(out_lol, self.test_lol)

        # check casting by defaults
        out_list = input_funcs.catch_list(self.test_dict, "test_list", [-1.0])
        out_lol = input_funcs.catch_list(self.test_dict, "test_lol", [[-1]], len_highest=len(self.test_lol))
        self.assertEqual(out_list, self.test_list)
        self.assertEqual(out_lol, self.test_lol)

        # test default assignment
        out_list = input_funcs.catch_list(self.test_dict, "no_list", self.test_list)
        out_lol = input_funcs.catch_list(self.test_dict, "no_lol", self.test_lol, len_highest=len(self.test_lol))
        self.assertEqual(out_list, self.test_list)
        self.assertEqual(out_lol, self.test_lol)

    def test_parse_line(self):

        out_int_key, out_int_value = input_funcs.parse_line("test_int = " + str(self.test_int) + "\n")
        out_float_key, out_float_value = input_funcs.parse_line("test_float = " + str(self.test_float) + "\n")
        out_bool_key, out_bool_value = input_funcs.parse_line("test_bool = " + str(self.test_bool) + "\n")
        out_str_key, out_str_value = input_funcs.parse_line('test_str = "' + self.test_str + '"\n')
        out_list_key, out_list_value = input_funcs.parse_line("test_list = " + str(self.test_list) + "\n")
        out_lol_key, out_lol_value = input_funcs.parse_line("test_lol = " + str(self.test_lol) + "\n")
        self.assertEqual(out_int_key, "test_int")
        self.assertEqual(out_int_value, self.test_int)
        self.assertEqual(out_float_key, "test_float")
        self.assertEqual(out_float_value, self.test_float)
        self.assertEqual(out_bool_key, "test_bool")
        self.assertEqual(out_bool_value, self.test_bool)
        self.assertEqual(out_str_key, "test_str")
        self.assertEqual(out_str_value, self.test_str)
        self.assertEqual(out_list_key, "test_list")
        self.assertEqual(out_list_value, self.test_list)
        self.assertEqual(out_lol_key, "test_lol")
        self.assertEqual(out_lol_value, self.test_lol)

    def test_read_input_file(self):

        out_dict = input_funcs.read_input_file("test_input_funcs.inp")
        self.assertEqual(out_dict["test_int"], self.test_int)
        self.assertEqual(out_dict["test_float"], self.test_float)
        self.assertEqual(out_dict["test_bool"], self.test_bool)
        self.assertEqual(out_dict["test_str"], self.test_str)

        # read_input_file automatically converts lists to NumPy arrays
        self.assertTrue(np.array_equal(out_dict["test_list"], self.test_list))
        self.assertTrue(np.array_equal(out_dict["test_lol"], self.test_lol))


if __name__ == "__main__":
    unittest.main()
