echo Running on $HOSTNAME
export CUDA_VISIBLE_DEVICES=${X_SGE_CUDA_DEVICE}
#echo $LD_LIBRARY_PATH

cd /home/dawna/tts/mw545/TorchTTS/espnet_py36/espnet/egs2/CUED_vctk/tts_integrated

# spk_model_name=cmp
# train_spk_dataset_type=cmp_binary_86_40
# inference_spk_dataset_type=cmp_binary_86_40

spk_model_name=sincnet
train_spk_dataset_type=wav_binary_3000_3000
inference_spk_dataset_type=wav_binary_3000_120

# spk_model_name=sincnet_4800
# train_spk_dataset_type=wav_binary_4800_4800
# inference_spk_dataset_type=wav_binary_4800_120

# spk_model_name=sinenet
# train_spk_dataset_type=wav_f_tau_vuv_binary_3000_3000
# inference_spk_dataset_type=wav_f_tau_vuv_binary_3000_120

# spk_model_name=sinenet_4800
# train_spk_dataset_type=wav_f_tau_vuv_binary_4800_4800
# inference_spk_dataset_type=wav_f_tau_vuv_binary_4800_120

train_spk_dataset_name=dynamic_5_seconds


# Training
if false; then
    ./run.sh --stage 6 --stop-stage 6 --use_spk_model true --train_config conf/train_${spk_model_name}_tacotron2.yaml --spk_model_name ${spk_model_name} --train_spk_dataset_name ${train_spk_dataset_name} --train_spk_dataset_type ${train_spk_dataset_type}
fi

# train_spk_dataset_name=dynamic_5_seconds
test_sets=eval1
inference_model=valid.loss.best

# Multi-second test
if true; then
# for num_seconds in 5 10 15 20 30 40 50; do
for num_seconds in 30; do
    for num_draw in {20..30}; do
    # for num_draw in 1; do
        inference_spk_dataset_name=same_${num_seconds}_seconds_per_speaker_draw_${num_draw}
        ./run.sh --stage 7 --stop-stage 7 --use_spk_model true --train_config conf/train_${spk_model_name}_tacotron2.yaml --spk_model_name ${spk_model_name} --inference_model ${inference_model}.pth --test_sets ${test_sets} --train_spk_dataset_name ${train_spk_dataset_name} --inference_spk_dataset_name ${inference_spk_dataset_name} --inference_spk_dataset_type ${inference_spk_dataset_type}  --inference_use_teacher_forcing true --inference_tag tf --generate_wav false --inference_keep_feats false --gpu_inference false --inference_nj 10
    done
done          
fi

# Speaker embedding generation
test_sets=spk_embed_eval1
if false; then
    inference_spk_dataset_name=speaker_draw_1_30_file_per_speaker_draw_30
    ./run.sh --stage 8 --stop-stage 8 --use_spk_model true --train_config conf/train_${spk_model_name}_tacotron2.yaml --spk_model_name ${spk_model_name} --inference_model ${inference_model}.pth --test_sets ${test_sets} --train_spk_dataset_name ${train_spk_dataset_name} --inference_spk_dataset_name ${inference_spk_dataset_name} --inference_spk_dataset_type ${inference_spk_dataset_type} 

fi

# Waveform Generation
test_sets=eval1
if false; then
    # for train_spk_dataset_name in dynamic_5_seconds_from_scratch; do
    for train_spk_dataset_name in dynamic_5_seconds; do
        # for inference_spk_dataset_name in same_5_seconds_per_speaker_draw_1 same_30_seconds_per_speaker_draw_1; do
        for inference_spk_dataset_name in same_30_seconds_per_speaker_draw_1; do
            ./run.sh --stage 7 --stop-stage 7 --use_spk_model true --train_config conf/train_${spk_model_name}_tacotron2.yaml --spk_model_name ${spk_model_name} --inference_model ${inference_model}.pth --test_sets ${test_sets} --train_spk_dataset_name ${train_spk_dataset_name} --inference_spk_dataset_name ${inference_spk_dataset_name} --inference_spk_dataset_type ${inference_spk_dataset_type}  --inference_use_teacher_forcing false --generate_wav false --gpu_inference false --inference_nj 10
            dataset_dir=exp/tts_train_${spk_model_name}_tacotron2_raw_phn_tacotron_g2p_en_no_space_${spk_model_name}_${train_spk_dataset_name}/decode_${inference_model}/${inference_spk_dataset_name}/${test_sets}
            checkpoint=pretrained_model/vctk_parallel_wavegan.v1.long/checkpoint-1000000steps.pkl
            . ./path.sh && parallel-wavegan-decode --checkpoint ${checkpoint} --feats-scp ${dataset_dir}/norm/feats.scp --outdir ${dataset_dir}/wav_pwg
        done
    done
fi


# Multi-file test
if false; then
for num_adapt_sentence in {17..30}; do
    for num_draw in {1..30}; do
        inference_spk_dataset_name=same_${num_adapt_sentence}_file_per_speaker_draw_${num_draw}
        ./run.sh --stage 7 --stop-stage 7 --use_spk_model true --train_config conf/train_${spk_model_name}_tacotron2.yaml --spk_model_name ${spk_model_name} --inference_model ${inference_model}.pth --test_sets ${test_sets} --train_spk_dataset_name ${train_spk_dataset_name} --inference_spk_dataset_name ${inference_spk_dataset_name} --inference_spk_dataset_type ${inference_spk_dataset_type}  --inference_use_teacher_forcing true --generate_wav false
    done
done          
fi

echo Finished "$(date +"%Y_%m_%d_%H_%M_%S")"

######################################################################################

if false; then
    for spk_model_name in cmp
    do
        for spk_dataset_name in random_1 random_5
        do
            # for inference_model in train.loss.best train.loss.ave valid.loss.best valid.loss.ave
            for inference_spk_dataset_name in random_1 random_5
            do
                test_sets=eval1
                inference_model=valid.loss.best
                if [ "${inference_spk_dataset_name}" = random_1 ]; then
                    inference_spk_dataset_type=cmp_binary_86_40_200
                fi
                if [ "${inference_spk_dataset_name}" = random_5 ]; then
                    inference_spk_dataset_type=cmp_binary_86_40_1000
                fi

                # Inference
                ./run.sh --stage 7 --stop-stage 7 --use_spk_model true --train_config conf/train_${spk_model_name}_tacotron2.yaml --spk_model_name ${spk_model_name} --inference_model ${inference_model}.pth --test_sets ${test_sets} --spk_dataset_name ${spk_dataset_name} --inference_spk_dataset_name ${inference_spk_dataset_name} --inference_spk_dataset_type ${inference_spk_dataset_type}
                # parallel-wavegan vocoder
                dataset_dir=exp/tts_train_${spk_model_name}_tacotron2_raw_phn_tacotron_g2p_en_no_space_${spk_model_name}_${spk_dataset_name}/decode_${inference_model}/${inference_spk_dataset_name}/${test_sets}
                checkpoint=pretrained_model/vctk_parallel_wavegan.v1.long/checkpoint-1000000steps.pkl
                . ./path.sh && parallel-wavegan-decode --checkpoint ${checkpoint} --feats-scp ${dataset_dir}/norm/feats.scp --outdir ${dataset_dir}/wav_pwg
            done
        done
    done
fi

# inference_spk_dataset_name=same
if false; then
    spk_model_name=cmp
    spk_dataset_name=random_1
    inference_model=valid.loss.best
    test_sets=eval1
    inference_spk_dataset_name=same
    inference_spk_dataset_type=cmp_binary_86_40_200
    ./run.sh --stage 7 --stop-stage 7 --use_spk_model true --train_config conf/train_${spk_model_name}_tacotron2.yaml --spk_model_name ${spk_model_name} --inference_model ${inference_model}.pth --test_sets ${test_sets} --spk_dataset_name ${spk_dataset_name} --inference_spk_dataset_name ${inference_spk_dataset_name} --inference_spk_dataset_type ${inference_spk_dataset_type} --inference_use_teacher_forcing true --inference_tag tf 
fi

if false; then
    # dataset_dir=exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_1/decode_valid.loss.best/same/eval1
    # dataset_dir=exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_5/tf/random_5/eval1
    dataset_dir=exp/tts_train_cmp_tacotron2_raw_phn_tacotron_g2p_en_no_space_cmp_random_5/decode_valid.loss.best/same_55_seconds_per_speaker/eval1
    checkpoint=pretrained_model/vctk_parallel_wavegan.v1.long/checkpoint-1000000steps.pkl
    . ./path.sh && parallel-wavegan-decode --checkpoint ${checkpoint} --feats-scp ${dataset_dir}/norm/feats.scp --outdir ${dataset_dir}/wav_pwg
fi

# ./run.sh --stage 9 --stop-stage 9 --use_spk_model true --train_config conf/train_${spk_model_name}_tacotron2.yaml --spk_model_name ${spk_model_name} --inference_model ${inference_model}.pth --test_sets ${test_sets} --spk_dataset_name ${spk_dataset_name} --inference_tag tf --inference_use_teacher_forcing true --gpu_inference true  --inference_spk_dataset_name ${inference_spk_dataset_name}

