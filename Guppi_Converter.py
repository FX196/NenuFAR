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

    def write_int8_to_file(self, output_file, header_gen, data_gen, directIO=True):
        """
        :param output_file: filename to write to
        :type output_file: str
        :param header_gen: generator that returnsa dictionary of values to specify in the header
        :type header_gen: object
        header_gen should have the same length as data_gen
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

        header_dict = next(header_gen)
        NPOL = int(header_dict["NPOL"]) // 2
        # number of polarizations, usually 4 for two polarizations of complex data
        # dividing by two to treat complex data as single polarization
        NBITS = int(header_dict["NBITS"])
        OBSNCHAN = int(header_dict["OBSNCHAN"])  # number of channels in the data
        BLOCSIZE = int(header_dict["BLOCSIZE"])

        assert NBITS == 8, "NBITS must be 8 for this data"

        NDIM = int(BLOCSIZE / (OBSNCHAN * (NPOL / 2) * (NBITS / 8))) // 2

        with open(output_file, "wb") as output:
            current_block = 0
            header = generate_header(header_dict)  # get formatted header from dictionary
            for data in data_gen:
                output.write(header)
                if self.verbose:
                    print("\rWriting Data Block %d" % current_block, end="")
                for channel_num in range(OBSNCHAN):
                    channel = data[channel_num]
                    output.write(channel.T.tobytes())

                # finished writing data block
                current_block += 1
                try:
                    header_dict = next(header_gen)
                except StopIteration:
                    print("Header Generator Depleted")
                    continue
                header = generate_header(header_dict)

    def test(self):
        print("success")


def get_sample_header(block_index=0):
    with open("sample_data/samp_header%d" % block_index, "rb") as file:
        header, h_length, header_dict = read_header(file)
    return header_dict


def get_dummy_header_gen(num_blocks=128):
    def dummy_header_gen():
        for i in range(num_blocks):
            yield get_sample_header(i)

    return dummy_header_gen()


def get_dummy_data_gen():
    def dummy_data_gen():
        np.random.seed(0)
        count = 0
        while count < 4:
            yield np.random.rand(64, 4, 128) * 50
            count += 1

    return dummy_data_gen()


def get_synthesized_data_gen(NDIM, NPOL, NCHAN, num_blocks=128):
    def synthesized_data_gen():
        x = np.arange(0, NDIM / 1000, 1 / 1000)
        pol_0_real = np.ones(NDIM) * 40
        pol_0_imag = np.ones(NDIM) * 20
        for _ in range(num_blocks):
            channels = np.zeros(shape=(NCHAN, NPOL * 2, NDIM), dtype=np.int8)
            for channel_ind in range(NCHAN):
                channel = generate_channel(NDIM, NPOL)
                channel[0] += pol_0_real
                inds = channel[0] >= 128
                channel[0][inds] = 127
                channel[1] += pol_0_imag
                inds = channel[1] >= 128
                channel[1][inds] = 127
                channels[channel_ind] = channel.astype(np.int8)
            yield channels

    return synthesized_data_gen()


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
        'int8-guppi': Data_Converter.write_int8_to_file
    }

    header_gen = get_dummy_header_gen()
    data_gen = get_synthesized_data_gen(524288, 2, 64)

    modes[args.conversion_mode](converter, "test_write.raw", header_gen, data_gen)
