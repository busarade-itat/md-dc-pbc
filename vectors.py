import re
import sys
import numpy


# String variable interpolation from local variable scope
# Source: https://dzone.com/articles/how-to-implement-string-interpolation-in-python-br
def interpolate(s):
    return s.format(**sys._getframe(1).f_locals)


class DomainElementFactory:
    VECTOR_SIZE = 1

    @staticmethod
    def positive_infinity():
        return DomainElementFactory._fill_with_number(float("inf"))

    @staticmethod
    def negative_infinity():
        return DomainElementFactory._fill_with_number(float("-inf"))

    @staticmethod
    def _fill_with_number(number):
        out = tuple()

        for _ in range(DomainElementFactory.VECTOR_SIZE):
            out = out + (number, )

        return RealVector(out)

    @staticmethod
    def _tuple_literal_is_correct(tuple_literal):
        reg_tuple_literal = "<(?:-?(?:inf|[0-9.]+),?)+>"

        if re.match(reg_tuple_literal, tuple_literal) is None:
            raise Exception(interpolate("{tuple_literal} is not valid TemporalVector string literal."))
        else:
            return True

    @staticmethod
    def parse_from_cpp(tuple_literal):
        if DomainElementFactory._tuple_literal_is_correct(tuple_literal):
            tuple_literal = tuple_literal.replace("<", "").replace(">", "")
            reg_number_literal = "(?:-?(?:inf|[0-9.]+))+"

            if re.match(reg_number_literal, tuple_literal) is not None:
                out = tuple()
                element_matches = re.finditer(reg_number_literal, tuple_literal)

                for match in element_matches:
                    if match[0] == "-inf":
                        out = out + (-float("inf"), )
                    elif match[0] == "inf":
                        out = out + (float("inf"), )
                    else:
                        out = out + (float(match[0]), )

                return RealVector(out)


class RealVector:
    def __init__(self, from_tuple):
        if self._is_tuple(from_tuple):
            self._vector_elements = from_tuple

    def __sub__(self, other):
        if self._is_realvector(other) and self._vectors_are_same_length(self, other):
            out = tuple()

            for self_element, other_element in zip(self._vector_elements, other._vector_elements):
                out = out + (self_element - other_element, )

            return RealVector(out)

    def __lt__(self, other):
        raise Exception("< got used somewhere")

    def is_inside_bounds(self, lower_bound, upper_bound):
        if self._vectors_are_same_length(self, lower_bound) and \
                self._vectors_are_same_length(self, upper_bound):
            for self_element, lb_element, ub_element in \
                    zip(self._vector_elements, lower_bound._vector_elements, upper_bound._vector_elements):
                if not self._is_in_interval(self_element, lb_element, ub_element):
                    return False

            return True

    def __neg__(self):
        out = tuple()

        for element in self._vector_elements:
            out = out + (-element, )

        return RealVector(out)

    def _is_tuple(self, an_object):
        if type(an_object) is not tuple:
            raise Exception("TemporalVector construction literal is not a tuple.")
        else:
            return True

    def _is_realvector(self, an_object):
        if not isinstance(an_object, RealVector):
            raise Exception("Supplied object is not a TemporalVector instance.")
        else:
            return True

    def _vectors_are_same_length(self, vec1, vec2):
        if len(vec1._vector_elements) != len(vec2._vector_elements):
            raise Exception("TemporalVectors to compare are not same size.")
        else:
            return True

    def _is_in_interval(self, item, lower_bound, upper_bound):
        if lower_bound > upper_bound:
            raise Exception("Lower bound is higher than upper bound for interval check.")
        else:
            return lower_bound <= item <= upper_bound

    def __str__(self):
        out = "<"

        for i in range(0, len(self._vector_elements)):
            out += numpy.format_float_positional(self._vector_elements[i]).rstrip(".")

            if i < len(self._vector_elements)-1:
                out += ","

        out += ">"

        return out

    def format_for_dcm_input(self):
        out = ""

        for i in range(0, len(self._vector_elements)):
            out += numpy.format_float_positional(self._vector_elements[i]).rstrip(".")

            if i < len(self._vector_elements) - 1:
                out += " "

        return out
