echo Running on $HOSTNAME
export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}
#echo $LD_LIBRARY_PATH
#./run.sh --use_xvector true
#./run.sh --stage 2 --stop-stage 3 --use_xvector true --train_config conf/train.yaml
#./run.sh --stage 20 --stop-stage 20 --use_xvector true --train_config conf/train.yaml
#./run.sh --stage 3 --stop-stage 5

./run.sh --stage 6 --stop-stage 7 --use_xvector true --train_config conf/train.yaml --spk_embed_name cmp
# ./run.sh --stage 5 --stop-stage 7 --use_xvector true --train_config conf/train.yaml --spk_embed_name sincnet
# ./run.sh --stage 5 --stop-stage 7 --use_xvector true --train_config conf/train.yaml --spk_embed_name xvector --spk_embed_type kaldi_ark


echo Finished "$(date +"%Y_%m_%d_%H_%M_%S")"
