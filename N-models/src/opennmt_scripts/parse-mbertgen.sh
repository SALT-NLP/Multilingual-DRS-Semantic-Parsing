#!/bin/bash
set -eu -o pipefail
### Arguments:
#		  $1 : config-file
#         $2 : input-file
#         $3 : output-file
#         $4 : model1
#         $5 : optional: model2 (model3, model4, etc)

source config/opennmt/default_config.sh #always load default settings from config file here
source $1 #first command line argument is the config file with specific settings -- it overrides settings in default_config.sh if added

#export PYTHONPATH=${GIT_HOME}DRS_parsing/evaluation/:${PYTHONPATH}

# Change if you want no logs
#disable_logs="-disable_logs"
#disable_logs=""

# NOTE: specifying multiple models can easily result in out-of-memory errors (on GPU)
# if you are parsing something that is not the small PMB data, be aware that this can happen

# Check if arguments are set -- $2 and $3 are input files and output files
DATA=$MAIN_FOLDER$WORKING


if [ ! -z "$2" ] && [ ! -z "$3" ] && [ ! -z "$4" ] && [ ! -z "$5" ] && [ ! -z "$6" ]; then
	# Get all models in a single param
	 models="${@:6}"
	# Do parsing here
    echo 'begin' > $3${res_ext}.counter.log 
    for f in $models
    do
        echo $f >> $3${res_ext}.counter.log 
        CUDA_VISIBLE_DEVICES=$gpuid fairseq-generate ${DATA}$DEST_EN \
  --path $f \
  --task translation_from_pretrained_mbert \
  --gen-subset test \
  -t $char_drs -s $sent$char_sent_ext \
  --bpe 'sentencepiece' --sentencepiece-model xlmr.base/sentencepiece.bpe.model \
  --remove-bpe 'sentencepiece' \
  --max-sentences 32 > $3.intermediate
  
  cat $3.intermediate | grep -P "^H" |sort -V |cut -f 3- > $3
         #CUDA_VISIBLE_DEVICES=$gpuid $TRANSLATE_LUA -src $2 -output $3 -model $f -beam_size $beam_size -max_length $max_sent_length $replace_unk -n_best $n_best $length_norm $coverage_norm -gpu 0
	# Postprocess
    echo $3
        python $POSTPROCESS_PY --input_file $3 --input_sent_file $2 --input_lex_file $4 --output_file $3$res_ext --var $var_rewrite $no_sep --sig_file $sig_file > $3${res_ext}.log
        ~/anaconda3/envs/py2/bin/python counter2.py -f1 $3$res_ext -f2 $5 -pr>> $3${res_ext}.counter.log
	#python $POSTPROCESS_PY --input_file $3 --output_file ${3}.noref --var $var_rewrite $no_sep --sig_file $sig_file --no_referee > ${3}.noref.log
    done
else
	echo "Usage: parse.sh CONFIG_FILE INPUT_FILE OUTPUT_FILE MODEL1 [MODEL2] [MODEL3] .."
fi
