import os


temp_dir = "temp"


class NewFileFormatConverter:
    CONVERTONLY_OUTPUT_DIR = "convertonly_output"
    CONVERSION_OUTPUT_DIR = temp_dir + "/dataset"
    REQUIRED_FILES = {
        "input": "input.csv",
        "satisfactory": "satisfactory.csv",
        "temperatures": "temperatures.csv"
    }
    OUTPUT_FORMAT_SEPARATOR = " "
    LABEL_OUTPUT_FILES = {
        "1": "pos.dat",
        "0": "neg.dat"
    }

    def __init__(self, dataset_dir, mock_input_files=None,
                 conversion_out_dir=CONVERSION_OUTPUT_DIR):
        self._dataset_dir = dataset_dir
        self._event_buffer = []
        self._temperature_buffer = []
        self._conversion_out_dir = conversion_out_dir

        # testing member variables
        # {input|satisfactory|temperatures} => array of input file lines
        self._mock_infiles = mock_input_files
        # {0|1} => array of output file lines
        self._mock_outfiles = {
            "0": [],
            "1": []
        }

    def convert_new_to_old(self):
        if self._check_file_presence() and self._check_same_file_length():
            vecsize = None

            input = self._open_file("input")
            satisfactory = self._open_file("satisfactory")
            temperatures = self._open_file("temperatures")

            for (input_line, temperatures_line, satisfactory_line) in \
                    zip(input, temperatures, satisfactory):
                input_list = self._parse_input_line(input_line)
                temperatures_list = self._parse_temperatures_line(temperatures_line)
                sequence_label = self._parse_satisfactory_line(satisfactory_line)

                if vecsize is None:
                    vecsize = self._infer_vector_size(input_list, temperatures_list)
                else:
                    line_vecsize = self._infer_vector_size(input_list, temperatures_list)

                    if line_vecsize != vecsize:
                        raise Exception("Fluctuating line length found in the dataset in the \""
                                        + self._dataset_dir + "\" directory.")
                    else:
                        vecsize = line_vecsize

                output_list = []

                for (event, i) in zip(input_list, range(0, len(input_list))):
                    output_list.append(event)

                    for j in range(i*vecsize, (i+1)*vecsize):
                        output_list.append(temperatures_list[j])

                self._write_output(sequence_label, output_list)

            return vecsize

    def _check_file_presence(self):
        if self._is_file("input") and self._is_file("satisfactory") and self._is_file("temperatures"):
            return True
        else:
            raise Exception("Incorrect file names in the \"" + self._dataset_dir + "\" directory.")

    def _check_same_file_length(self):
        if self._get_line_count("input") == self._get_line_count("satisfactory") == self._get_line_count("temperatures"):
            return True
        else:
            raise Exception("Files in the \"" + self._dataset_dir + "\" directory have differing line count.")

    def _get_line_count(self, filekey):
        return sum(1 for line in self._open_file(filekey))

    def _infer_vector_size(self, input_list, temperatures_list):
        inferred_length = len(temperatures_list) / len(input_list)

        if round(inferred_length) != inferred_length:
            raise Exception("input.csv/temperatures.csv item counts do not match to consistent vector size.")
        else:
            return int(inferred_length)

    def _parse_input_line(self, input_line):
        return list(input_line.rstrip())

    def _parse_satisfactory_line(self, input_line):
        satisfactory_char = input_line.rstrip()

        if satisfactory_char != "0" and satisfactory_char != "1":
            raise Exception("satisfactory.csv file may contain only 0/1 characters.")
        else:
            return satisfactory_char

    def _parse_temperatures_line(self, input_line):
        value_list = input_line.rstrip().split(",")

        for value in value_list:
            try:
                float(value)
            except ValueError:
                raise Exception("temperatures.csv does not contain properly formatted numeric values.")

        return value_list

    def _convert_output_list_to_str(self, lst):
        return NewFileFormatConverter.OUTPUT_FORMAT_SEPARATOR.join(lst)

    def _get_input_filename(self, filekey):
        return self._dataset_dir + "/" + NewFileFormatConverter.REQUIRED_FILES[filekey]

    def _get_output_filename(self, label):
        return self._conversion_out_dir + "/" + NewFileFormatConverter.LABEL_OUTPUT_FILES[label]

    def _open_file(self, filekey):
        if self._mock_infiles is not None:
            if filekey == "input":
                return self._mock_infiles["input"]
            elif filekey == "satisfactory":
                return self._mock_infiles["satisfactory"]
            elif filekey == "temperatures":
                return self._mock_infiles["temperatures"]
        else:
            return open(self._get_input_filename(filekey), "r")

    def _is_file(self, filekey):
        if self._mock_infiles is not None:
            return self._mock_infiles[filekey] is not None
        else:
            return os.path.isfile(self._get_input_filename(filekey))

    def _write_output(self, filekey, output_list):
        output_file_path = self._get_output_filename(filekey)

        if self._mock_infiles is not None:
            self._mock_outfiles[filekey].append(self._convert_output_list_to_str(output_list))
        else:
            output_file_dir = self._conversion_out_dir

            if not os.path.exists(output_file_dir):
                os.makedirs(output_file_dir)

            fopen_flags = "a"

            if not os.path.isfile(output_file_path):
                fopen_flags = "w"

            with open(output_file_path, fopen_flags) as outfile:
                outfile.writelines([self._convert_output_list_to_str(output_list) + "\n"])
