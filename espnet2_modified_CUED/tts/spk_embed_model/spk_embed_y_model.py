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

