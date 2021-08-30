#!/bin/bash
set -eu -o pipefail
# Train an OpenNMT model based on a configuration file added as $1

source config/opennmt/default_config.sh #always load default settings from config file here
source $1 #first command line argument is the config file with specific settings -- it overrides settings in default_config.sh if added



# Set variables
START=1
DEP_DICT=/nethome/jyang690/DRSparsing/SemanticParsingPMB/PMB2/gold/dep.dict
PRETRAIN=xlmr.base/model.pt # fix if you moved the downloaded checkpoint
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
DATA=$MAIN_FOLDER$WORKING
  
# Training over (possibly) multiple runs
for run in $(eval echo "{$START..$num_runs}")
do
	# First create extra directory for the models if num_runs > 1
	if [[ (( $num_runs -gt 1)) ]]; then
		mod_folder="model_$run"
		mkdir -p $MAIN_FOLDER$MODELS$mod_folder
	else
		mod_folder=""
	fi

	model_folder="$MAIN_FOLDER$MODELS${mod_folder}/"
	random_seed=$RANDOM # Use a different random seed every time, but can also fix a seed here
	echo "Start training run ${run}/$num_runs"
    : 'CUDA_VISIBLE_DEVICES=$gpuid fairseq-train ${DATA}$DEST_EN \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbertgen_base --layernorm-embedding \
  --task translation_from_pretrained_mbert \
  --source-lang $sent$char_sent_ext --target-lang $char_drs \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-eps 1e-06 --adam-betas 0.9, 0.999 \
  --lr-scheduler polynomial_decay --lr 3e-5 --min-lr -1 --warmup-updates 2500 --total-num-update 40000 --end-learning-rate 8e-6\
  --dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 --update-freq 2 \
  --save-interval 1 --save-interval-updates 1000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed $random_seed --log-format simple --log-interval 2 \
  --restore-file $PRETRAIN \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --ddp-backend no_c10d \
  --save-dir $model_folder --memory-efficient-fp16 \
  --max-source-positions 512 --max-target-positions 512'
  
    CUDA_VISIBLE_DEVICES=$gpuid fairseq-train ${DATA}$DEST_EN \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbertgen_base --layernorm-embedding \
  --task translation_from_pretrained_mbert --depdict ${DEP_DICT} \
  --source-lang $sent$char_sent_ext --target-lang $char_drs \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam_encdec --adam-eps 1e-06 --adam-betas '(0.9, 0.999)' \
  --lr-scheduler polynomial_decay_encdec --min-lr -1 --lr-enc 2e-5 --warmup-updates-enc 5000 --total-num-update-enc 40000 --lr-dec 5e-5 --warmup-updates-dec 2500 --total-num-update-dec 40000 --end-learning-rate 8e-6 \
  --dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 --update-freq 2 \
  --save-interval 1 --save-interval-updates 1000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed $random_seed --log-format simple --log-interval 2 \
  --restore-file $PRETRAIN \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --ddp-backend no_c10d \
  --save-dir $model_folder --memory-efficient-fp16 \
  --max-source-positions 512 --max-target-positions 512
  
	#CUDA_VISIBLE_DEVICES=$gpuid $TRAIN_LUA -data $MAIN_FOLDER$VOCAB$VOCAB_NAME -src_word_vec_size $src_word_vec_size -tgt_word_vec_size $tgt_word_vec_size -layers $layers -rnn_size $rnn_size -encoder_type $model_type -decoder_type $model_type $transformer_ff $heads $pos_enc  -save_model "$MAIN_FOLDER$MODELS${mod_folder}/model" -log_file $MAIN_FOLDER$LOG$log_file_train -dropout $dropout $bridge -global_attention $global_attention $copy_att -report_every $report_every -train_steps $train_steps -save_checkpoint_steps $save_checkpoint_steps -valid_steps $valid_steps -batch_size $batch_size -batch_type $batch_type -optim $optim -learning_rate $learning_rate -max_grad_norm $max_grad_norm $train_from $pre_word_vecs_enc $pre_word_vecs_dec -seed $random_seed -gpu_ranks 0 -keep_checkpoint 10 -start_decay_steps $start_decay_steps -decay_steps $decay_steps $fix_word_vec $feat_vec
done
