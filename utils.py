import numpy as np
from matplotlib import pyplot as plt


def read_bit(file):
    return int.from_bytes(file.read(1), byteorder="big", signed=True)
    # byteorder ???


def generate_2d_normal(size):
    return np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=size).T * 20


def read_header(file):
    header = ""
    h_length = 0
    header_dict = {}
    s = ""

    while s[:3] != "END":
        h_length += 80
        s = file.read(80)
        s = s.decode("utf-8")
        header += s + "\n"
        if not s[:3] == "END":
            key, value = s.split("= ")
            key, value = key.strip(), value.strip()

            header_dict[key] = value
    if "DIRECTIO" in header_dict.keys() and "1" in header_dict["DIRECTIO"]:
        # remove padding bits due to DIRECTIO
        file.read(512 - h_length % 512)
    return header, h_length, header_dict


def bleed_data(file, header_dict):
    channel_size = int(
        int(header_dict["BLOCSIZE"]) / (int(header_dict["OBSNCHAN"]) * 2 * int(header_dict["NBITS"]) / 8))
    n_chan = int(header_dict["OBSNCHAN"])
    for i in range(n_chan):
        file.read(int(header_dict["BLOCSIZE"]) // int(header_dict["OBSNCHAN"]))


def generate_channel(ndim, npol):
    """
    generate channel data with gaussian noise
    :param ndim: length of the channel
    :param npol: number of polarizations; can be either 1 or 2
    :return:
    """
    channel = np.zeros(shape=(npol * 2, ndim))
    for i in range(npol):
        pol_data = generate_2d_normal(ndim)
        channel[i * 2:i * 2 + 2] = pol_data[:]
    return channel


def generate_header(header_dict):
    """
    generate a header in the form of a byte-stream from a python dictionary
    with padding bytes in case DIRECTIO is set to 1
    :param header_dict: python dictionary containing the header key-value pairs
    :return: header in the form of a byte-stream
    """
    header = ""
    for key, value in header_dict.items():
        if len(key) > 8:
            raise Exception("Header key should be no more than 8 characters")
        if len(str(value)) > 70:
            print(value)
            raise Exception(
                "Header value should be no more than 70 characters")
        key = key.ljust(8)
        if type(value) == int or type(value) == float:
            value = str(value) + " " * 50
            value = value.rjust(70)
        else:
            value = value.ljust(70)
        header += key + "= " + value
    header += "END".ljust(80)
    if "DIRECTIO" in header_dict.keys() and header_dict["DIRECTIO"] == 1:
        header += " " * (512 - (len(header) % 512))
    return header.encode("utf-8")


def read_channels(file, header):
    n_chan = int(header['OBSNCHAN'])
    n_pol = int(header['NPOL'])
    n_samples = int(int(header['BLOCSIZE']) / n_chan / n_pol)
    n_bit = int(header['NBITS'])
    print((n_chan, n_samples, n_pol))

    data = np.fromfile(file, count=int(header['BLOCSIZE']), dtype='int8')

    data = data.reshape((n_chan, n_samples, n_pol))

    return data


def plot_channel(file, header_dict):
    channel = read_channels(file, header_dict)[0]

    pol_0 = channel[:, 0] + 1j * channel[:, 1]
    pol_1 = channel[:, 2] + 1j * channel[:, 3]

    plt.figure(figsize=(10, 5))
    plt.plot(pol_0)
    plt.title("Channel Polarization 0 Raw")

    spectrum = np.fft.fftshift(np.fft.fft(pol_0))
    spectrum = 10. * np.log10(abs(spectrum) ** 2)
    plt.figure(figsize=(10, 5))
    plt.plot(spectrum)
    plt.title("Channel Polarization 0 Freqs")

    plt.figure(figsize=(10, 5))
    plt.plot(pol_1)
    plt.title("Channel Polarization 1 Raw")

    spectrum = np.fft.fftshift(np.fft.fft(pol_1))
    spectrum = 10. * np.log10(abs(spectrum) ** 2)
    plt.figure(figsize=(10, 5))
    plt.plot(spectrum)
    plt.title("Channel Polarization 1 Freqs")


def plot_channels(num_channels, file, channel_size):
    global count
    channels = read_channels(file, channel_size)
    for i in range(num_channels):
        channel = channels[i]

        channel = np.array(channel)

        pol_0 = channel[:, 0] + 1j * channel[:, 1]
        pol_1 = channel[:, 2] + 1j * channel[:, 3]

        plt.figure(figsize=(10, 5))
        plt.plot(pol_0)
        plt.title("Channel " + str(count) + " Polarization 0 Raw")

        spectrum = np.fft.fftshift(np.fft.fft(pol_0))
        spectrum = 10. * np.log10(abs(spectrum) ** 2)
        plt.figure(figsize=(10, 5))
        plt.plot(spectrum)
        plt.title("Channel " + str(count) + " Polarization 0 Freqs")

        plt.figure(figsize=(10, 5))
        plt.plot(pol_1)
        plt.title("Channel " + str(count) + " Polarization 1 Raw")

        spectrum = np.fft.fftshift(np.fft.fft(pol_1))
        spectrum = 10. * np.log10(abs(spectrum) ** 2)
        plt.figure(figsize=(10, 5))
        plt.plot(spectrum)
        plt.title("Channel " + str(count) + " Polarization 1 Freqs")

        count += 1


def plot_channel_bartlett(file, header_dict, num_windows):
    channel = read_channels(file, header_dict)[0]
    channel = np.array(channel)
    res_size = channel.shape[0]//num_windows
    res_0 = np.zeros(res_size)
    res_1 = np.zeros(res_size)

    for i in range(num_windows):
        window, channel = channel[:res_size], channel[res_size:]
        pol_0 = window[:, 0] + 1j * window[:, 1]
        pol_1 = window[:, 2] + 1j * window[:, 3]

        spectrum_0 = np.fft.fftshift(np.fft.fft(pol_0))
        spectrum_0 = 10. * np.log10(abs(spectrum_0) ** 2)
        res_0 += spectrum_0

        spectrum_1 = np.fft.fftshift(np.fft.fft(pol_1))
        spectrum_1 = 10. * np.log10(abs(spectrum_1) ** 2)
        res_1 += spectrum_1

    res_0 /= num_windows
    res_1 /= num_windows

    plt.figure(figsize=(10, 5))
    plt.plot(res_0)
    plt.title("Polarization 0 Freqs with " + str(num_windows) + " windows")

    plt.figure(figsize=(10, 5))
    plt.plot(res_1)
    plt.title("Polarization 1 Freqs with " + str(num_windows) + " windows")
