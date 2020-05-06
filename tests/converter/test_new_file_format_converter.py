import unittest
import sys

sys.path.insert(0, '..')

from new_file_format_converter import *

MOCK_INFILES_CORRECT = {
    "input": [
        "ABCD",
        "BCDA",
        "CDAB",
        "DABC"
    ],
    "temperatures": [
        "15.0,15.0,23.0,23.0,29.0,29.0,37.0,37.0",
        "15.0,15.0,16.0,16.0,28.0,28.0,29.0,29.0",
        "15.0,15.0,16.0,16.0,25.0,25.0,26.0,26.0",
        "15.0,15.0,21.0,21.0,22.0,22.0,28.0,28.0"
    ],
    "satisfactory": [
        "1",
        "1",
        "0",
        "0"
    ]
}

MOCK_INFILES_DIFFERENT_LENGTH = {
    "input": [
        "ABCD",
        "BCDA",
        "CDAB",
        "DABC"
    ],
    "temperatures": [
        "15.0,15.0,23.0,23.0,29.0,29.0,37.0,37.0",
        "15.0,15.0,16.0,16.0,28.0,28.0,29.0,29.0",
        "15.0,15.0,16.0,16.0,25.0,25.0,26.0,26.0"
    ],
    "satisfactory": [
        "1",
        "1",
        "0",
        "0"
    ]
}

MOCK_INFILES_MISSING = {
    "input": [
        "ABCD",
        "BCDA",
        "CDAB",
        "DABC"
    ],
    "temperatures": [
        "15.0,15.0,23.0,23.0,29.0,29.0,37.0,37.0",
        "15.0,15.0,16.0,16.0,28.0,28.0,29.0,29.0",
        "15.0,15.0,16.0,16.0,25.0,25.0,26.0,26.0",
        "15.0,15.0,21.0,21.0,22.0,22.0,28.0,28.0"
    ]
}

MOCK_INFILES_FLUCTUATING = {
    "input": [
        "ABCD",
        "BCD",
        "CDAB",
        "DABC"
    ],
    "temperatures": [
        "15.0,15.0,23.0,23.0,29.0,29.0,37.0",
        "15.0,15.0,16.0,16.0,28.0,28.0,29.0,29.0",
        "15.0,15.0,16.0,16.0,25.0,25.0",
        "15.0,15.0,21.0,21.0,22.0,22.0,28.0,28.0"
    ],
    "satisfactory": [
        "1",
        "1",
        "0",
        "0"
    ]
}

class TestNewFileFormatConverter(unittest.TestCase):
    def test_check_same_file_length(self):
        conv = NewFileFormatConverter("", MOCK_INFILES_DIFFERENT_LENGTH)

        # it raises an exception when input files have differing line count
        with self.assertRaises(Exception):
            conv.convert_new_to_old()

    def test_check_file_presence(self):
        conv = NewFileFormatConverter("", MOCK_INFILES_MISSING)

        # it raises an exception when some input files are missing in the dataset directory
        with self.assertRaises(Exception):
            conv.convert_new_to_old()

    def test_fluctuating_line_length(self):
        conv = NewFileFormatConverter("", MOCK_INFILES_FLUCTUATING)

        # it raises an exception when there is fluctuating number of items in file lines
        with self.assertRaises(Exception):
            conv.convert_new_to_old()

    def test_parse_input_line(self):
        conv = NewFileFormatConverter("")

        # it parses a line correctly
        self.assertTrue(conv._parse_input_line("ABCD") == ["A", "B", "C", "D"])

        # it parses a line correctly when in a file with LF line endings
        self.assertTrue(conv._parse_input_line("ABCD\n") == ["A", "B", "C", "D"])

        # it parses a line correctly when in a file with CRLF line endings
        self.assertTrue(conv._parse_input_line("ABCD\r\n") == ["A", "B", "C", "D"])

        # it parses a line with trailing whitespace correctly
        self.assertTrue(conv._parse_input_line("ABCD     ") == ["A", "B", "C", "D"])

        # it does not raise an exception when provided with whitespace or metacharacters
        self.assertTrue(conv._parse_input_line("A ,D") == ["A", " ", ",", "D"])

    def test_parse_satisfactory_line(self):
        conv = NewFileFormatConverter("")

        # it parses a line correctly
        self.assertTrue(conv._parse_satisfactory_line("1") == "1")

        # it parses a line correctly when in a file with LF line endings
        self.assertTrue(conv._parse_satisfactory_line("0\n") == "0")

        # it parses a line correctly when in a file with CRLF line endings
        self.assertTrue(conv._parse_satisfactory_line("0\r\n") == "0")

        # it raises an exception for a line not containing the "0" or "1" character
        with self.assertRaises(Exception):
            conv._parse_satisfactory_line("L")

        # it raises an exception for a line containing multiple characters
        with self.assertRaises(Exception):
            conv._parse_satisfactory_line("01")

    def test_parse_temperatures_line(self):
        conv = NewFileFormatConverter("")

        # it parses a line correctly
        self.assertTrue(conv._parse_temperatures_line("1,4.2,4.000005,0.0") == ["1", "4.2", "4.000005", "0.0"])

        # it parses a line correctly when in a file with LF line endings
        self.assertTrue(conv._parse_temperatures_line("1,4.2,4.000005,0.0\n") == ["1", "4.2", "4.000005", "0.0"])

        # it parses a line correctly when in a file with CRLF line endings
        self.assertTrue(conv._parse_temperatures_line("1,4.2,4.000005,0.0\r\n") == ["1", "4.2", "4.000005", "0.0"])

        # it raises an exception for a line not containing a correctly formatted float number
        with self.assertRaises(Exception):
            conv._parse_temperatures_line("1,4.2,4R,0.0")

    def test_convert_output_list_to_str(self):
        conv = NewFileFormatConverter("")

        # it serializes a list of arbitrary stringified values correctly
        self.assertTrue(conv._convert_output_list_to_str(["1.000000005", "L"]) == "1.000000005 L")

        # it serializes an empty list to empty line
        self.assertTrue(conv._convert_output_list_to_str([]) == "")

    def test_get_input_filename(self):
        conv = NewFileFormatConverter("indir")

        # it produces correct input file paths for known file types
        self.assertTrue(conv._get_input_filename("satisfactory") == "indir/satisfactory.csv")
        self.assertTrue(conv._get_input_filename("input") == "indir/input.csv")
        self.assertTrue(conv._get_input_filename("temperatures") == "indir/temperatures.csv")

        # it raises an exception for unknown file type
        with self.assertRaises(Exception):
            conv._get_input_filename("xxx")

    def test_get_output_filename(self):
        conv = NewFileFormatConverter("")

        old_out_dir = NewFileFormatConverter.CONVERSION_OUTPUT_DIR
        NewFileFormatConverter.CONVERSION_OUTPUT_DIR = "outdir"

        # it produces correct output file paths for known sequence labels
        self.assertTrue(conv._get_output_filename("0") == "temp/dataset/neg.dat")
        self.assertTrue(conv._get_output_filename("1") == "temp/dataset/pos.dat")

        # it raises an exception for unknown sequence label
        with self.assertRaises(Exception):
            conv._get_output_filename("R")

        NewFileFormatConverter.CONVERSION_OUTPUT_DIR = old_out_dir

    def test_successful_conversion(self):
        conv = NewFileFormatConverter("", MOCK_INFILES_CORRECT)

        # it infers vector size correctly
        self.assertTrue(conv.convert_new_to_old() == 2)

        # it converts positive sequences to old format correctly
        self.assertTrue(
            conv._mock_outfiles["1"][0] == "A 15.0 15.0 B 23.0 23.0 C 29.0 29.0 D 37.0 37.0" and
            conv._mock_outfiles["1"][1] == "B 15.0 15.0 C 16.0 16.0 D 28.0 28.0 A 29.0 29.0"
        )

        # it converts negative sequences to old format correctly
        self.assertTrue(
            conv._mock_outfiles["0"][0] == "C 15.0 15.0 D 16.0 16.0 A 25.0 25.0 B 26.0 26.0" and
            conv._mock_outfiles["0"][1] == "D 15.0 15.0 A 21.0 21.0 B 22.0 22.0 C 28.0 28.0"
        )

if __name__ == '__main__':
    unittest.main()
