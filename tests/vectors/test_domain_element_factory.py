import unittest
import sys

sys.path.insert(0, '..')

from vectors import DomainElementFactory


class TestTemporalElementFactory(unittest.TestCase):
    def test_positive_infinity(self):
        DomainElementFactory.VECTOR_SIZE = 1

        # it produces a vector of positive infinity values properly
        # it reacts to TEMPORAL_VECTOR_SIZE value changes
        self.assertTrue(
            DomainElementFactory.positive_infinity()
            ._vector_elements == (float("inf"), )
        )

    def test_negative_infinity(self):
        DomainElementFactory.VECTOR_SIZE = 2

        # it produces a vector of negative infinity values properly
        # it reacts to TEMPORAL_VECTOR_SIZE value changes
        self.assertTrue(
            DomainElementFactory.negative_infinity()
            ._vector_elements == (float("-inf"), float("-inf"))
        )

    def test_tuple_literal_is_correct(self):
        tl1 = "<1,2>"
        tl2 = "<-inf,inf,-inf>"
        tl3 = "<0.00005,1.5,0.0000000005>"
        tl4 = "(0.00005,1.5,0.0000000005>"
        tl5 = "(69,>"
        tl6 = "<1.2e3,10>"
        tl7 = "<1, 2>"
        tl8 = "<1,abc>"

        # it marks well-formed tuple literals as correct
        self.assertTrue(DomainElementFactory._tuple_literal_is_correct(tl1))
        self.assertTrue(DomainElementFactory._tuple_literal_is_correct(tl2))
        self.assertTrue(DomainElementFactory._tuple_literal_is_correct(tl3))

        # it raises an exception for malformed tuple literals
        with self.assertRaises(Exception):
            DomainElementFactory._tuple_literal_is_correct(tl4)

        with self.assertRaises(Exception):
            DomainElementFactory._tuple_literal_is_correct(tl5)

        with self.assertRaises(Exception):
            DomainElementFactory._tuple_literal_is_correct(tl6)

        with self.assertRaises(Exception):
            DomainElementFactory._tuple_literal_is_correct(tl7)

        with self.assertRaises(Exception):
            DomainElementFactory._tuple_literal_is_correct(tl8)

    def test_parse_from_cpp(self):
        tl1 = "<1,-2>"
        tl2 = "<-inf,inf,-inf>"
        tl3 = "<0.00005,1.5,0.0000000005>"

        # it parses tuple literals containing whole numbers correctly
        self.assertTrue(
            DomainElementFactory.parse_from_cpp(tl1)
            ._vector_elements == (1, -2)
        )

        # it parses tuple literals containing +/-inf values correctly
        self.assertTrue(
            DomainElementFactory.parse_from_cpp(tl2)
            ._vector_elements == (float("-inf"), float("inf"), float("-inf"))
        )

        # it parses tuple literals containing floating point numbers correctly
        self.assertTrue(
            DomainElementFactory.parse_from_cpp(tl3)
            ._vector_elements == (0.00005, 1.5, 0.0000000005)
        )


if __name__ == '__main__':
    unittest.main()
