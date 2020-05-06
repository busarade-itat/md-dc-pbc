import unittest
import sys

sys.path.insert(0, '..')

from vectors import RealVector


class TestRealVector(unittest.TestCase):
    def test_constructor(self):
        tup = (1, 2, 3)

        # it constructs a RealVector from a tuple correctly
        tv = RealVector(tup)
        self.assertTrue(tv._vector_elements == tup)

    def test_is_in_interval(self):
        tv = RealVector(tuple())

        # the closed interval check works correctly
        self.assertTrue(tv._is_in_interval(1, 0, 2))
        self.assertFalse(tv._is_in_interval(3, 0, 2))

        # the closed interval check works correctly for edge values
        self.assertTrue(tv._is_in_interval(0, 0, 2))

        # the closed interval check works correctly for +/-inf bound values
        self.assertTrue(tv._is_in_interval(0, float("-inf"), float("inf")))

        # the interval check raises an exception when interval bounds are not properly ordered
        with self.assertRaises(Exception):
            tv._is_in_interval(0, float("inf"), float("-inf"))

        # the interval check does not raise an exception for lower bound equal to upper bound
        self.assertTrue(tv._is_in_interval(0, 0, 0))

    def test_subtraction(self):
        tv1 = RealVector((1, 2, 3))
        tv2 = RealVector((1, 1, 1))

        tvres = tv1 - tv2

        # it subtracts two RealVectors correctly
        self.assertTrue(tvres._vector_elements == (0, 1, 2))

    def test_is_inside_bounds(self):
        tvb1 = RealVector((1, 2, 3))
        tvb2 = RealVector((3, 4, 5))
        tvb3 = RealVector((float("inf"), 4, 5))

        tvt1 = RealVector((2, 3, 4))  # completely inside R(tvb1, tvb2)
        tvt2 = RealVector((1, 2, 3))  # touching R(tvb1, tvb2) bounds
        tvt3 = RealVector((4, 4, 4))  # first coord is not inside R(tvb1, tvb2)
        tvt4 = RealVector((4, 5, 6))  # all coords are outside R(tvb1, tvb2)

        # it performs the hyperrectangle test correctly
        self.assertTrue(tvt1.is_inside_bounds(tvb1, tvb2))

        # it performs the hyperrectangle test correctly for points on hyperrectangle bounds
        self.assertTrue(tvt2.is_inside_bounds(tvb1, tvb2))

        # it performs the hyperrectangle test correctly for hyperrectangle bounds containing inf/-inf
        self.assertTrue(tvt3.is_inside_bounds(tvb1, tvb3))

        # it performs the hyperrectangle test correctly for some coordinates outside of hyperrectangle bounds
        self.assertFalse(tvt3.is_inside_bounds(tvb1, tvb2))

        # it performs the hyperrectangle test correctly for all coordinates outside of hyperrectangle bounds
        self.assertFalse(tvt4.is_inside_bounds(tvb1, tvb2))

    def test_unary_minus(self):
        tv1 = RealVector((1, 2, 3))
        tv2 = RealVector((0, 0, 0))

        # the unary minus operation negates RealVector elements correctly
        self.assertTrue((-tv1)._vector_elements == (-1, -2, -3))
        self.assertTrue((-tv2)._vector_elements == (0, 0, 0))

    def test_is_tuple(self):
        tv = RealVector(tuple())

        # _is_tuple correctly marks a tuple literal as tuple
        self.assertTrue(tv._is_tuple(tuple()))

        # _is_tuple raises an exception when supplied a dictionary
        with self.assertRaises(Exception):
            tv._is_tuple({})

        # _is_tuple raises an exception when supplied an instance of RealVector
        with self.assertRaises(Exception):
            tv._is_tuple(RealVector(tuple()))

    def test_is_realvector(self):
        tv = RealVector(tuple())

        # _is_realvector correctly marks a RealVector object as RealVector instance
        self.assertTrue(tv._is_realvector(RealVector(tuple())))

        # _is_realvector raises an exception when supplied a tuple
        with self.assertRaises(Exception):
            tv._is_realvector(tuple())

        # _is_realvector raises an exception when supplied a dictionary
        with self.assertRaises(Exception):
            tv._is_realvector({})

    def test_vectors_are_same_length(self):
        tv = RealVector(tuple())

        tv1 = RealVector((1, 2, 3))
        tv2 = RealVector((2, 2, 3))
        tv3 = RealVector((2, 3))

        # _vectors_are_same_length works correctly for two RealVectors of the same length
        self.assertTrue(tv._vectors_are_same_length(tv1, tv2))

        # _vectors_are_same_length raises an exception for two RealVectors of different length
        with self.assertRaises(Exception):
            tv._vectors_are_same_length(tv1, tv3)

    def test_formatting(self):
        tv1 = RealVector((1, 2, 3))
        tv2 = RealVector((float("inf"), float("inf")))
        tv3 = RealVector((float("-inf"),))
        tv4 = RealVector((0.00005, 1.4, 0.0000000005))

        # it formats a RealVector containing whole numbers correctly
        self.assertTrue(tv1.format_for_dcm_input() == "1 2 3")
        self.assertTrue(str(tv1) == "<1,2,3>")

        # it formats RealVectors containing +/-inf values correctly
        self.assertTrue(tv2.format_for_dcm_input() == "inf inf")
        self.assertTrue(str(tv2) == "<inf,inf>")

        self.assertTrue(tv3.format_for_dcm_input() == "-inf")
        self.assertTrue(str(tv3) == "<-inf>")

        # it formats a RealVector containing floating point numbers correctly
        # it formats a RealVector containing very small floating
        # point numbers without using scientific notation
        self.assertTrue(tv4.format_for_dcm_input() == "0.00005 1.4 0.0000000005")
        self.assertTrue(str(tv4) == "<0.00005,1.4,0.0000000005>")

if __name__ == '__main__':
    unittest.main()
