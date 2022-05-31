import os, sys
sys.path.append("/home/dawna/tts/mw545/TorchDV/tools/merlin_cued_mw545")

# from espnet2_modified_CUED.merlin_cued_mw545.nn_torch.torch_models import Build_DV_Y_model
# from espnet2_modified_CUED.merlin_cued_mw545.cfg_main import configuration
# from espnet2_modified_CUED.merlin_cued_mw545.exp_mw545.exp_dv_config import dv_y_configuration

from nn_torch.torch_models import Build_DV_Y_model
from run_24kHz import configuration

# class dv_y_cmp_configuration(dv_y_configuration):
#     def __init__(self, cfg):
#         super().__init__(cfg)

#         self.retrain_model = False
#         self.learning_rate  = 0.0001
#         # self.prev_nnets_file_name = ''
#         self.python_script_name = os.path.realpath(__file__)
#         # self.data_dir_mode = 'data' # Use scratch for speed up

#         # cmp input configuration
#         self.y_feat_name   = 'cmp'
#         self.init_cmp_data()
#         self.out_feat_list = ['mgc', 'lf0', 'bap']
#         self.update_cmp_dim()

#         self.dv_dim = 512
#         # self.input_data_dim['S'] = 1 # For computing GPU requirement
#         self.nn_layer_config_list = [
#             {'type':'LReLU', 'size':256*2, 'dropout_p':0, 'layer_norm':True},
#             {'type':'LReLU', 'size':256*2, 'dropout_p':0, 'layer_norm':True},
#             {'type':'Linear', 'size':self.dv_dim, 'dropout_p':0, 'layer_norm':True}
#         ]

#         # self.gpu_id = 'cpu'
#         self.gpu_id = 0
#         self.auto_complete(cfg, cache_files=False)

def Build_spk_embed_y_model(spk_model_name):
    cfg = configuration(cache_files=False)

    if spk_model_name == 'cmp':
        from exp_mw545.exp_dv_cmp_baseline import dv_y_cmp_configuration
        dv_y_cfg = dv_y_cmp_configuration(cfg, cache_files=False)
        model = Build_DV_Y_model(dv_y_cfg)
        return model.nn_model

    if spk_model_name == 'sincnet':
        from exp_mw545.exp_dv_wav_sincnet import dv_y_wav_sincnet_configuration
        dv_y_cfg = dv_y_wav_sincnet_configuration(cfg, cache_files=False)
        model = Build_DV_Y_model(dv_y_cfg)
        model.torch_initialisation()
        prev_nnets_file_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_sincnet/dvy_wav_lr1E-04_fpu40_Sin60_LRe512L_LRe512L_Lin512L_DV512S10T3000/Model'
        model.load_nn_model(prev_nnets_file_name)
        return model.nn_model

    if spk_model_name == 'sincnet_4800':
        from exp_mw545.exp_dv_wav_sincnet import dv_y_wav_sincnet_configuration
        dv_y_cfg = dv_y_wav_sincnet_configuration(cfg, cache_files=False)
        dv_y_cfg.input_data_dim['T_B'] = int(0.2 * dv_y_cfg.cfg.wav_sr)
        dv_y_cfg.input_data_dim['B_stride'] = dv_y_cfg.input_data_dim['T_B']
        dv_y_cfg.update_wav_dim()
        model = Build_DV_Y_model(dv_y_cfg)
        return model.nn_model

    if spk_model_name == 'sinenet':
        from exp_mw545.exp_dv_wav_sinenet_v0 import dv_y_wav_sinenet_configuration
        dv_y_cfg = dv_y_wav_sinenet_configuration(cfg, cache_files=False)
        model = Build_DV_Y_model(dv_y_cfg)
        model.torch_initialisation()
        # if dv_y_cfg.prev_nnets_file_name is None:
        #     prev_nnets_file_name = dv_y_cfg.nnets_file_name
        # else:
        #     prev_nnets_file_name = dv_y_cfg.prev_nnets_file_name
        prev_nnets_file_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v0/dvy_wav_lr1E-04_fpu40_Sin80af64ks0.5T_LRe512L_LRe512L_Lin512L_DV512S10T3000TM240/Model'
        model.load_nn_model(prev_nnets_file_name)
        return model.nn_model

    if spk_model_name == 'sinenet_4800':
        from exp_mw545.exp_dv_wav_sinenet_v0 import dv_y_wav_sinenet_configuration
        dv_y_cfg = dv_y_wav_sinenet_configuration(cfg, cache_files=False)
        dv_y_cfg.input_data_dim['T_B'] = int(0.2 * dv_y_cfg.cfg.wav_sr)
        dv_y_cfg.input_data_dim['B_stride'] = dv_y_cfg.input_data_dim['T_B']
        dv_y_cfg.update_wav_dim()

        dv_y_cfg.nn_layer_config_list[2] = {'type': 'Sinenet_V0', 'size':60, 'num_freq':64, 'k_space':0.5, 'dropout_p':0, 'use_f': 'D', 'use_tau': 'D', 'inc_a':True, 'k_train':True, 'batch_norm':False}
        model = Build_DV_Y_model(dv_y_cfg)
        # model.torch_initialisation()
        # if dv_y_cfg.prev_nnets_file_name is None:
        #     prev_nnets_file_name = dv_y_cfg.nnets_file_name
        # else:
        #     prev_nnets_file_name = dv_y_cfg.prev_nnets_file_name
        # model.load_nn_model(dv_y_cfg.nnets_file_name)
        return model.nn_model

    if spk_model_name == 'sinenet_v1':
        from exp_mw545.exp_dv_wav_sinenet_v1 import dv_y_wav_sinenet_configuration
        dv_y_cfg = dv_y_wav_sinenet_configuration(cfg, cache_files=False)
        model = Build_DV_Y_model(dv_y_cfg)
        model.torch_initialisation()
        # if dv_y_cfg.prev_nnets_file_name is None:
        #     prev_nnets_file_name = dv_y_cfg.nnets_file_name
        # else:
        #     prev_nnets_file_name = dv_y_cfg.prev_nnets_file_name
        prev_nnets_file_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v1/dvy_wav_lr1E-04_fpu40_Sin80f64ks0.5T_LRe512L_LRe512L_Lin512L_DV512S10T3000TM240/Model'
        model.load_nn_model(prev_nnets_file_name)
        return model.nn_model

    if spk_model_name == 'sinenet_v2':
        from exp_mw545.exp_dv_wav_sinenet_v2 import dv_y_wav_sinenet_configuration
        dv_y_cfg = dv_y_wav_sinenet_configuration(cfg, cache_files=False)
        model = Build_DV_Y_model(dv_y_cfg)
        model.torch_initialisation()
        # if dv_y_cfg.prev_nnets_file_name is None:
        #     prev_nnets_file_name = dv_y_cfg.nnets_file_name
        # else:
        #     prev_nnets_file_name = dv_y_cfg.prev_nnets_file_name
        prev_nnets_file_name = '/home/dawna/tts/mw545/TorchDV/dv_wav_sinenet_v2/dvy_wav_lr1E-04_fpu40_Sin80f64ks0.5T_LRe512L_LRe512L_Lin512L_DV512S10T3000TM240/Model'
        model.load_nn_model(prev_nnets_file_name)
        return model.nn_model

