#!/usr/bin/python3

import sys, getopt, tqdm, argparse
from utils import *
import numpy as np


# write to file
# with open("pure_noise_unscaled.0000.raw", "wb") as file:
#     for i in tqdm(range(128)):
#         header, header_dict = get_sample_header(source_file)
#         d = {"NPOL": 4, "OBSNCHAN": 64, "NBITS": 8,
#          "BLOCSIZE": 134217728, "DIRECTIO": 1,
#         "OBSERVER": "'Yuhong Chen'", "SRC_NAME":"'SYNTH'"}
#         for key, value in d.items():
#             header_dict[key] = value
#         d = header_dict
#         header = generate_header(d)
#         file.write(header)
#         for j in range(64):
#             channel = generate_channel(NDIM, NPOL)
#             channel = np.array(channel, dtype = np.int8)
#             #print("Writing Channel {0} of Data Block {1}".format(j, i), end="\r", flush=True)
#             file.write(channel.T.tobytes())
#         #print("Finished Writing Data Block {0}     ".format(i), end="\n\r", flush=True)
#     print("Done")

class Data_Converter:

    def __init__(self, input_file, output_file, verbose):
        self.input_file = input_file
        self.output_file = output_file
        self.verbose = verbose

        print(self.__dict__)

    def write_int8_to_file(self, output_file, header_dict, data_gen, directIO=True):
        """
        :param output_file: filename to write to
        :type output_file: str
        :param header_dict: a dictionary of values to specify in the header
        :type header_dict: dict
        :param data_gen: a generator that outputs data as int8 numpy arrays
        :type data_gen: object
        the output of data_gen should be a 3d int8 array,
        with channel number, polarization, and index as the three indices
        with data = data_gen.next()
        data[c][p][i] will be the voltage sample at the i'th time step of the p'th polarization of the c'th channel
        :param directIO: flag for directIO
        :type directIO: bool
        :return:
        """
        NPOL = int(header_dict["NPOL"])  # number of polarizations, double for complex data
        OBSNCHAN = int(header_dict["OBSNCHAN"])  # number of channels in the data

        assert int(header_dict["NBITS"]) == 8, "NBITS must be 8 for this data"

        header = generate_header(header_dict)  # get formatted header from dictionary

        with open(output_file, "wb") as output:
            current_block = 0
            for data in data_gen:
                if self.verbose:
                    print("\rWriting Data Block %d" % current_block)
                for channel_num in range(OBSNCHAN):
                    channel = data[channel_num]
                    output.write(channel.T.tobytes())

                # finished writing data block
                current_block += 1

    def test(self):
        print("success")


def get_dummy_data_gen():
    def dummy_data_gen():
        np.random.seed(0)
        count = 0
        while count < 4:
            yield np.random.rand(1, 1, 128) * 50
            count += 1

    return dummy_data_gen()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str, required=True,
                        help="Path to the Input File")
    parser.add_argument("-o", "--output-file", type=str, required=True,
                        help="Path to the Output File")
    parser.add_argument("-m", "--conversion-mode", type=str, help="Mode for Data Conversion; e.g. int8 to Guppi RAW",
                        default='int8-guppi')
    parser.add_argument("-v", "--verbose", action='count', default=0, help="Set Verbosity")
    args = parser.parse_args()

    converter = Data_Converter(input_file=args.input_file, output_file=args.output_file, verbose=args.verbose > 0)

    modes = {
        'int8-guppi': Data_Converter.test
    }

    modes[args.conversion_mode](converter)
