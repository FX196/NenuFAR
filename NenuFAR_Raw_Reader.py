#!/usr/bin/env python3

###############################################################################
#
# Copyright (C) 2019
# Station de Radioastronomie de Nançay,
# Observatoire de Paris, PSL Research University, CNRS, Univ. Orléans, OSUC,
# 18330 Nançay, France
#
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
# Author: Cedric Viou (Cedric.Viou@obs-nancay.fr)
#
# Description:
# Lib that can read nenufar-raw output files
#
###############################################################################


from pathlib import Path
import numpy as np
import re
import datetime as dt


class raw:
    def __init__(self, obs_dir, verbose=False):
        self.fnames_info = sorted(list(obs_dir.glob("*.spectra.info")))
        self.fnames_log = sorted(list(obs_dir.glob("*.spectra.log")))
        self.fnames_data = sorted(list(obs_dir.glob("*.spectra")))
        self.fnames_raw = sorted(list(obs_dir.glob("*.raw")))

        self.raw = []
        self.verbose = verbose

        for lane_idx, (fn_raw) in enumerate(self.fnames_raw):
            if verbose:
                print("Opening %s" % (fn_raw.name))
            dt_header, dt_block = self.infer_data_structure(fn_raw)
            raw = self.open_raw(fn_raw, dt_header, dt_block)
            self.raw.append(raw)
            if self.verbose:
                pass
                # print("{} -> nbchan={}, fftlen={}, nfft2int={}".format(fn_info.name,
                #                                  self.nof_beamlets[lane_idx],
                #                                  self.fftlen[lane_idx],
                #                                  self.nfft2int[lane_idx],
                #                                  ))

    def infer_data_structure(self, fn_raw):
        """Parse raw file
        """
        dt_header = np.dtype([('nobpb', 'int32'),
                              ('nb_samples', 'int32'),
                              ('bytespersample', 'int32'),
                              ])

        fd_raw = fn_raw.open('rb')
        header = np.frombuffer(fd_raw.read(dt_header.itemsize),
                               count=1,
                               dtype=dt_header,
                               )[0]
        fd_raw.close()

        nobpb = header['nobpb']
        nb_samples = header['nb_samples']
        bytespersample = header['bytespersample']
        bytes_in = bytespersample * nb_samples * nobpb
        if self.verbose:
            print(dt_header)
            print(header)

        dt_block = np.dtype([('eisb', 'uint64'),
                             ('tsb', 'uint64'),
                             ('bsnb', 'uint64'),
                             ('data', 'int8', (bytes_in,)),
                             ])

        return dt_header, dt_block

    def open_raw(self, fn_raw, dt_header, dt_block):
        """Return a memory mapped file array"""
        # tmp = np.memmap(fn_raw.as_posix(),
        #                 dtype='int8',
        #                 mode='r',
        #                 offset=dt_header.itemsize,
        #                 )
        # # Reinterpret an integer number of dt_block-length samples as dt_block
        # nof_blocks = tmp.size * tmp.itemsize // (dt_block.itemsize)
        # data = tmp[: nof_blocks * dt_block.itemsize].view(dt_block)

        data = np.memmap(fn_raw.as_posix(),
                         dtype=dt_block,
                         mode='r',
                         offset=dt_header.itemsize,
                         )
        return data


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.dates as md
    import datetime as dt
    import sys

    plot_auto_corr = True
    plot_cross_corr = True

    block_idx = 0
    if len(sys.argv) > 1:
        block_idx = int(sys.argv[1])

    chan_start, chan_stop = 112, 175
    f0 = (200 + chan_start) * 200.0 / 1024
    f1 = (200 + chan_stop) * 200.0 / 1024

    obs_dir, block_idx = Path('/databf2/nenufar-tf/B1919+21_TRACKING_20190319_080534'), 25
    # obs_dir, block_idx = Path('/databf2/nenufar-tf/20190319_112500_20190319_113100_B2217+47'), 0

    assert obs_dir.exists(), "File not found.  Check path, or try running on a nancep node"

    my_spectra = raw(obs_dir,
                     verbose=True,
                     )

    dat = my_spectra.raw[0]['data']
    nof_blocks = dat.shape[0]
    nobpb = 192
    fftlen = 32
    nof_polcpx = 4
    nfft = dat.shape[1] // nobpb // fftlen // nof_polcpx

    chan_start *= fftlen
    chan_stop *= fftlen

    for block_num in range(block_idx, block_idx + 1):

        max_bsn = 200e6 / 1024
        ts = my_spectra.raw[0][block_num:block_num + 2]['tsb'].astype("double") + \
             my_spectra.raw[0][block_num:block_num + 2]['bsnb'].astype("double") / max_bsn
        dates = list(dt.datetime.fromtimestamp(t) for t in ts)
        t0, t1 = md.date2num(dates)
        dates = list(t.isoformat() for t in dates)

        tmp = dat[block_num]
        tmp.shape = (nfft, fftlen, nobpb, nof_polcpx)
        tmp = tmp.astype('float32').view('complex64')
        tmp = tmp.transpose((3, 0, 2, 1))
        TMP = np.fft.fft(tmp, axis=-1)
        TMP = np.fft.fftshift(TMP, axes=-1)
        TMP.shape = (2, nfft, nobpb * fftlen)

        do_acc = False

        if plot_auto_corr:
            fig1, axs = plt.subplots(1, 2, sharex=True, sharey=True, )  # figsize=(30,15))
            fig1.suptitle("%s - %s" % tuple(dates))
            for polar, ax in enumerate(axs):
                data_plot = TMP[polar, :, chan_start:chan_stop]
                data_plot = data_plot.real ** 2 + data_plot.imag ** 2
                if do_acc:
                    data_plot.shape = (272, 10, -1)
                    data_plot = data_plot.mean(axis=(1,))
                    v = (-3, 3)
                else:
                    v = (0, 10)
                data_plot = 10 * np.log10(data_plot + 1e-10)
                ref = data_plot.mean(axis=(0))
                data_plot -= ref
                ax.imshow(data_plot,
                          extent=(f0, f1, t1, t0),
                          interpolation='nearest',
                          origin='upper',
                          vmin=v[0], vmax=v[1],
                          cmap="Greys_r"
                          )
                ax.axis('tight')
                ax.set_title(("XX", "YY")[polar])
                xfmt = md.DateFormatter('%Y-%m-%d\n%H:%M:%S\n+0.%f')
                ax.yaxis.set_major_formatter(xfmt)

        if plot_cross_corr:
            XC = TMP[0, :, chan_start:chan_stop] * TMP[1, :, chan_start:chan_stop].conj()
            if do_acc:
                XC.shape = (272, 10, -1)
                XC = XC.mean(axis=(1,))
                v = (0, 5)
            else:
                v = (0, 10)

            fig2, axs2 = plt.subplots(1, 2, sharex=True, sharey=True)

            data_plot = np.angle(XC)
            axs2[0].imshow(data_plot,
                           extent=(f0, f1, t1, t0),
                           interpolation='nearest',
                           origin='upper',
                           vmin=-np.pi,
                           vmax=np.pi,
                           # cmap="Greys_r",
                           cmap="hsv",
                           )
            axs2[0].axis('tight')
            axs2[0].set_title("angle(XY)")
            xfmt = md.DateFormatter('%Y-%m-%d\n%H:%M:%S\n+0.%f')
            axs2[0].yaxis.set_major_formatter(xfmt)

            data_plot = np.sqrt(XC.real ** 2 + XC.imag ** 2)
            data_plot = 10 * np.log10(data_plot + 1e-10)
            ref = data_plot.mean(axis=(0))
            data_plot -= ref

            axs2[1].imshow(data_plot,
                           extent=(f0, f1, t1, t0),
                           interpolation='nearest',
                           origin='upper',
                           vmin=v[0], vmax=v[1],
                           cmap="Greys_r"
                           )
            axs2[1].axis('tight')
            axs2[1].set_title("10 log10 |XY|")
            xfmt = md.DateFormatter('%Y-%m-%d\n%H:%M:%S\n+0.%f')
            axs2[1].yaxis.set_major_formatter(xfmt)

    plt.show()
