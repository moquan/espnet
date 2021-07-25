import collections.abc
from pathlib import Path
from typing import Union

import numpy
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text

class cmp_reader(collections.abc.Mapping):
    """Reader class for a scp file of cmp file.

    Examples:
        key1 /some/path/a.cmp
        key2 /some/path/b.cmp
        key3 /some/path/c.cmp
        key4 /some/path/d.cmp
        ...

        >>> reader = NpyScpReader('cmp.scp')
        >>> array = reader['key1']

    """

    def __init__(self, fname: Union[Path, str], loader_type: str):
        assert check_argument_types()
        self.fname = Path(fname)
        self.data = read_2column_text(fname)

        self.feat_dim, self.window_size, self.total_length = map(int, loader_type[len("cmp_binary_") :].split("_"))
        self.num_windows = self.total_length - self.window_size + 1

    def get_path(self, key):
        return self.data[key]

    def __getitem__(self, key) -> numpy.ndarray:
        p = self.data[key]
        data = numpy.fromfile(p, dtype=numpy.float32)
        data = data.reshape((-1, self.feat_dim))
        return self.make_cmp_BD_from_data(data)

    def make_cmp_BD_from_data(self, cmp_data):
        frame_number = cmp_data.shape[0]
        extra_file_len = frame_number - self.total_length - 10
        if extra_file_len < 0:
            # pad if necessary
            cmp_data = numpy.concatenate((cmp_data, numpy.tile(cmp_data[[-1], :], (0-extra_file_len,1))), axis=0)
            extra_file_len = 0

        extra_file_len_ratio = numpy.random.rand()
        start_sample_no_sil = int(extra_file_len_ratio * extra_file_len+1)
        start_frame_include_sil = start_sample_no_sil + 5

        cmp_BD = numpy.zeros((self.num_windows, self.feat_dim * self.window_size))

        for b in range(self.num_windows):
            n_start = start_frame_include_sil + b
            n_end   = n_start + self.window_size
            cmp_TD  = cmp_data[n_start:n_end]
            cmp_D   = cmp_TD.reshape(-1)
            cmp_BD[b] = cmp_D

        return cmp_BD

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()