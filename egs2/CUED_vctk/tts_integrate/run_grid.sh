echo Running on $HOSTNAME
export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}
#echo $LD_LIBRARY_PATH

spk_model_name=cmp
# ./run.sh --stage 6 --stop-stage 7 --use_spk_model true --train_config conf/train_${spk_model_name}_tacotron2.yaml --spk_model_name ${spk_model_name}
./run.sh --stage 7 --stop-stage 7 --use_spk_model true --train_config conf/train_${spk_model_name}_tacotron2.yaml --spk_model_name ${spk_model_name} --inference_model train.loss.best.pth --test_sets  eval1
#./run.sh --stage 6 --stop-stage 6 --use_spk_model true --train_config conf/train.yaml --spk_model_name cmp
#./run.sh --stage 6 --stop-stage 7 --use_spk_model true --train_config conf/train.yaml --spk_model_name ${spk_model_name}



if false; then
    for spk_embed_name in gst
    do
        for inference_model in train.loss.best train.loss.ave valid.loss.best valid.loss.ave
        do
            test_sets=eval1
            # ./run.sh --stage 7 --stop-stage 7 --use_xvector true --train_config conf/train.yaml --spk_embed_name ${spk_embed_name} --inference_model ${inference_model}.pth --test_sets ${test_sets}
        done
    done
fi

if false; then
    for spk_model_name in gst
    do
        for pth_name in train.loss.best train.loss.ave valid.loss.best valid.loss.ave
        do
            dataset_dir=exp/tts_train_raw_phn_tacotron_g2p_en_no_space_${spk_model_name}/decode_${pth_name}/eval1
            checkpoint=pretrained_model/vctk_parallel_wavegan.v1.long/checkpoint-1000000steps.pkl
            # . ./path.sh && parallel-wavegan-decode --checkpoint ${checkpoint} --feats-scp ${dataset_dir}/norm/feats.scp --outdir ${dataset_dir}/wav_pwg
        done
    done
fi

echo Finished "$(date +"%Y_%m_%d_%H_%M_%S")"
