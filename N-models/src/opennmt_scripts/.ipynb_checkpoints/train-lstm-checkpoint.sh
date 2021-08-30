#!/bin/bash
set -eu -o pipefail
# Train an OpenNMT model based on a configuration file added as $1

source config/opennmt/default_config.sh #always load default settings from config file here
source $1 #first command line argument is the config file with specific settings -- it overrides settings in default_config.sh if added

if [[ -z ${both_vocab+x} ]]; then
	both_vocab=$MAIN_FOLDER$VOCAB$VOCAB_NAME$ext_voc
fi

# First get word embeddings for this vocab
if [[ $representation = "word" || $representation = 'char_word' ]]; then
	src_embed="$MAIN_FOLDER$embed.enc.pt"
	tgt_embed="$MAIN_FOLDER$embed.dec.pt"

	if [[ ! -f $src_embed ]] && [[ ! -f $tgt_embed ]]; then #only create if not exists already
		mkdir -p $MAIN_FOLDER$embed
		$EMBED_LUA -emb_file_both $embed_file -type $embed_type -dict_file $both_vocab -output_file $MAIN_FOLDER$embed  
	else
		echo "Embed file already exists, skip creating $src_embed and $tgt_embed"
	fi

	# Overwrite (possible) values of config file
	pre_word_vecs_enc="-pre_word_vecs_enc $src_embed"
	pre_word_vecs_dec="-pre_word_vecs_dec $tgt_embed"
fi

# Set variables
START=1

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
    CUDA_VISIBLE_DEVICES=$gpuid $TRAIN_LUA -data $MAIN_FOLDER$VOCAB$VOCAB_NAME -src_word_vec_size $src_word_vec_size -tgt_word_vec_size $tgt_word_vec_size -layers $layers -rnn_size $rnn_size -rnn_type $rnn_type -save_model "$MAIN_FOLDER$MODELS${mod_folder}/model" -log_file $MAIN_FOLDER$LOG$log_file_train -dropout $dropout $bridge -encoder_type $encoder_type -global_attention $global_attention -report_every $report_every -train_steps $train_steps -save_checkpoint_steps $save_checkpoint_steps -valid_steps $valid_steps -start_decay_steps $start_decay_steps -decay_steps $decay_steps -batch_size $batch_size -batch_type $batch_type -optim $optim -learning_rate $learning_rate -max_grad_norm $max_grad_norm -learning_rate_decay $learning_rate_decay $train_from $pre_word_vecs_enc $pre_word_vecs_dec -seed $random_seed $copy_att  -gpu_ranks 0 $fix_word_vec $feat_vec
done
