import collections.abc
from pathlib import Path
from typing import Union

import os, sys, numpy, pickle
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text

sys.path.append("/home/dawna/tts/mw545/TorchDV/tools/merlin_cued_mw545")
from run_24kHz import configuration
from frontend_mw545.modules import File_List_Selecter, List_Random_Loader, read_file_list
from frontend_mw545.data_io import Data_File_IO

from frontend_mw545.data_loader import Build_dv_TTS_selecter

        
class data_reader_base(collections.abc.Mapping):
    """
    Abstract Reader class for a scp file
    """

    def __init__(self, fname: Union[Path, str], loader_type: str):
        assert check_argument_types()
        self.fname = Path(fname)
        self.data = read_2column_text(fname)

        self.cfg = configuration(cache_files=False)
        self.dv_tts_selector = Build_dv_TTS_selecter(self.cfg, dv_y_cfg=None)

    def __getitem__(self, key) -> numpy.ndarray:
        # This is the main method to write for all data readers
        p = self.data[key]
        return self.input_string_handler(p)


    def input_string_handler(self, p):
        '''
        Possible string formats:
        1. speaker_id (mostly unused)
            p001_001 p001
        2. file_id
            p001_001 p001_060
        3. one string, some rules; contains 3 parts when split by '_'
            p001_001 p001_5_seconds
        4. file_ids, split by ' '
            p001_001 p001_060 p001_064 p001_065
        return:
            BD data
        '''
        file_list = p.split(' ')
        if len(file_list) == 1:
            num_p_parts = len(p.split('_'))
            if num_p_parts == 1:
                # e.g. p001, use speaker method
                return self.speaker_id_handler(p)
            elif num_p_parts == 2:
                # e.g. p001_002, use file embedding
                return self.file_id_str_handler(p)
            elif num_p_parts == 3:
                # e.g. p001_5_seconds
                file_id_str = self.draw_data_list(p)
                return self.file_id_str_handler(file_id_str)
        else:
            file_id_str = '|'.join(file_list)
            return self.file_id_str_handler(file_id_str)

    def speaker_id_handler(self, speaker_id):
        pass

    def file_id_str_handler(self, file_id_str):
        '''
        Input: file_ids joined by '|'
        '''
        pass

    def draw_data_list(self, p):
        '''
        Input: string, has 3 parts when split by '_'
            p001_5_seconds      Draw no more than 5 seconds from p001
            p001_5_files        Draw 5 files from p001
        Output:
            string, file_ids joined by '|'
        '''
        speaker_id, n, draw_rule = p.split('_')

        if draw_rule == 'files':
            # Draw n files
            n = int(n)
            file_list_str = self.dv_tts_selector.draw_n_files(speaker_id, 'SR', n)
        elif draw_rule == 'seconds':
            n = float(n)
            file_list_str = self.dv_tts_selector.draw_n_seconds(speaker_id, 'SR', n)
        return file_list_str

    '''
    Below are methods from original code, maybe just leave them
    '''
    def get_path(self, key):
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()



class spk_embed_reader(data_reader_base):
    """
    loader_type: spk_embed_%i % dimension
    """

    def __init__(self, fname: Union[Path, str], loader_type: str):
        super().__init__(fname, loader_type)

        self.feat_dim = int(loader_type.split('_')[-1])

        print("input file name is %s" % fname)

        self.dv_dir = self.get_dv_dir_from_fname(fname)
        self.load_dv_data()

    def get_dv_dir_from_fname(self, fname):
        '''
        e.g. 
        fname = 'dump/xvector/cmp_5s/tr_no_dev/dynamic_5_seconds.scp'
        return: 'dump/xvector/cmp_5s'
        '''
        f_list = fname.split('/')
        f_list_useful = f_list[:-2]
        return '/'.join(f_list_useful)

    def load_dv_data(self):
        dv_spk_dict_file = os.path.join(self.dv_dir, 'dv_spk_dict.dat')
        self.dv_spk_dict = pickle.load(open(dv_spk_dict_file, 'rb'))
        dv_file_dict_file = os.path.join(self.dv_dir, 'dv_file_dict.dat')
        self.dv_file_dict = pickle.load(open(dv_file_dict_file, 'rb'))

    def speaker_id_handler(self, speaker_id):
        return self.dv_spk_dict[speaker_id]

    def file_id_str_handler(self, file_id_str):
        file_list = file_id_str.split('|')
        total_frames = 0.
        dv_value = numpy.zeros(self.feat_dim)
        for f in file_list:
            num_frames, dv_value_f = self.dv_file_dict[f]
            total_frames += num_frames
            dv_value = dv_value + dv_value_f * num_frames
        return dv_value / total_frames


class cmp_reader(data_reader_base):
    """
    Reader class for a scp file of cmp file.
    loader_type: cmp_binary_%i_%i % dimension, window_size
    """

    def __init__(self, fname: Union[Path, str], loader_type: str):
        super().__init__(fname, loader_type)

        self.feat_dim, self.window_size = map(int, loader_type[len("cmp_binary_") :].split("_"))

        from exp_mw545.exp_dv_cmp_baseline import dv_y_cmp_configuration
        from frontend_mw545.data_loader import Build_dv_y_cmp_data_loader_Multi_Speaker
        self.cfg = configuration(cache_files=False)
        self.dv_y_cfg = dv_y_cmp_configuration(self.cfg, cache_files=False)
        self.data_loader = Build_dv_y_cmp_data_loader_Multi_Speaker(self.cfg, self.dv_y_cfg)

        assert self.dv_y_cfg.input_data_dim['T_B'] == self.window_size
        assert self.dv_y_cfg.input_data_dim['D'] == (self.window_size * self.feat_dim)

    def file_id_str_handler(self, file_id_str):
        BD_data, B, start_sample_number = self.data_loader.make_BD_data(file_id_str, start_sample_number=0)
        return BD_data