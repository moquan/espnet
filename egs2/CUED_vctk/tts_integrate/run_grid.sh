echo Running on $HOSTNAME
export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}
#echo $LD_LIBRARY_PATH
./run.sh --stage 5 --stop-stage 7 --train_config conf/train.yaml --spk_model_name gst --use_spk_model true

echo Finished "$(date +"%Y_%m_%d_%H_%M_%S")"
