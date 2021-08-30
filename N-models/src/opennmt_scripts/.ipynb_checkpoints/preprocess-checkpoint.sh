#!/bin/bash
set -eu -o pipefail
# Preprocess script for the DRS parsing seq2seq experiments

source config/opennmt/default_config.sh #always load default settings from config file here
source $1 #first command line argument is the config file with specific settings -- it overrides settings in default_config.sh if added

# First create directories (if not there yet)
mkdir -p $MAIN_FOLDER
mkdir -p $MAIN_FOLDER$WORKING
mkdir -p $MAIN_FOLDER$WORKING$TRAIN
mkdir -p $MAIN_FOLDER$WORKING$DEV
mkdir -p $MAIN_FOLDER$WORKING$TEST
mkdir -p $MAIN_FOLDER$WORKING$DE
mkdir -p $MAIN_FOLDER$WORKING$IT
mkdir -p $MAIN_FOLDER$WORKING$NL
mkdir -p $MAIN_FOLDER$MODELS
mkdir -p $MAIN_FOLDER$OUTPUT
mkdir -p $MAIN_FOLDER$OUTPUT$DEV
mkdir -p $MAIN_FOLDER$VOCAB
mkdir -p $MAIN_FOLDER$EVAL
mkdir -p $MAIN_FOLDER$LOG

# Copy the training files to the working directories (sentences and DRSs)
cp $GOLD_FOLDER$gold_train $MAIN_FOLDER$WORKING$TRAIN
cp $GOLD_FOLDER$gold_dev $MAIN_FOLDER$WORKING$DEV
cp $GOLD_FOLDER$gold_test $MAIN_FOLDER$WORKING$TEST
cp $GOLD_DE_FOLDER$gold_de $MAIN_FOLDER$WORKING$DE
cp $GOLD_IT_FOLDER$gold_it $MAIN_FOLDER$WORKING$IT
cp $GOLD_NL_FOLDER$gold_nl $MAIN_FOLDER$WORKING$NL

cp $GOLD_FOLDER$gold_train$sent_ext $MAIN_FOLDER$WORKING$TRAIN
cp $GOLD_FOLDER$gold_dev$sent_ext $MAIN_FOLDER$WORKING$DEV
cp $GOLD_FOLDER$gold_test$sent_ext $MAIN_FOLDER$WORKING$TEST
cp $GOLD_DE_FOLDER$gold_de$sent_ext $MAIN_FOLDER$WORKING$DE
cp $GOLD_IT_FOLDER$gold_it$sent_ext $MAIN_FOLDER$WORKING$IT
cp $GOLD_NL_FOLDER$gold_nl$sent_ext $MAIN_FOLDER$WORKING$NL


cp $GOLD_FOLDER$gold_train$lex_ext $MAIN_FOLDER$WORKING$TRAIN
cp $GOLD_FOLDER$gold_dev$lex_ext $MAIN_FOLDER$WORKING$DEV
cp $GOLD_FOLDER$gold_test$lex_ext $MAIN_FOLDER$WORKING$TEST
cp $GOLD_DE_FOLDER$gold_de$lex_ext $MAIN_FOLDER$WORKING$DE
cp $GOLD_IT_FOLDER$gold_it$lex_ext $MAIN_FOLDER$WORKING$IT
cp $GOLD_NL_FOLDER$gold_nl$lex_ext $MAIN_FOLDER$WORKING$NL

# Do Python preprocessing to put files in character-level format, for train/dev
# Remove ill-formed DRSs from train set
python $PREPROCESS_PYTHON --input_file $MAIN_FOLDER$WORKING$TRAIN$gold_train --sentence_file $MAIN_FOLDER$WORKING$TRAIN$gold_train$sent_ext --casing $casing --representation $representation --variables $var_rewrite --char_drs_ext $char_drs_ext --char_sent_ext $char_sent_ext --var_drs_ext $var_drs_ext
python $PREPROCESS_PYTHON --input_file $MAIN_FOLDER$WORKING$DEV$gold_dev --sentence_file $MAIN_FOLDER$WORKING$DEV$gold_dev$sent_ext --casing $casing --representation $representation --variables $var_rewrite --char_drs_ext $char_drs_ext --char_sent_ext $char_sent_ext --var_drs_ext $var_drs_ext
python $PREPROCESS_PYTHON --input_file $MAIN_FOLDER$WORKING$TEST$gold_test --sentence_file $MAIN_FOLDER$WORKING$TEST$gold_test$sent_ext --casing $casing --representation $representation --variables $var_rewrite --char_drs_ext $char_drs_ext --char_sent_ext $char_sent_ext --var_drs_ext $var_drs_ext
python $PREPROCESS_PYTHON --input_file $MAIN_FOLDER$WORKING$DE$gold_de --sentence_file $MAIN_FOLDER$WORKING$DE$gold_de$sent_ext --casing $casing --representation $representation --variables $var_rewrite --char_drs_ext $char_drs_ext --char_sent_ext $char_sent_ext --var_drs_ext $var_drs_ext
python $PREPROCESS_PYTHON --input_file $MAIN_FOLDER$WORKING$IT$gold_it --sentence_file $MAIN_FOLDER$WORKING$IT$gold_it$sent_ext --casing $casing --representation $representation --variables $var_rewrite --char_drs_ext $char_drs_ext --char_sent_ext $char_sent_ext --var_drs_ext $var_drs_ext
python $PREPROCESS_PYTHON --input_file $MAIN_FOLDER$WORKING$NL$gold_nl --sentence_file $MAIN_FOLDER$WORKING$NL$gold_nl$sent_ext --casing $casing --representation $representation --variables $var_rewrite --char_drs_ext $char_drs_ext --char_sent_ext $char_sent_ext --var_drs_ext $var_drs_ext

# Then do OpenNMT preprocessing to create the vocabulary files
train_src=$MAIN_FOLDER$WORKING$TRAIN$gold_train$sent_ext$char_sent_ext
train_tgt=$MAIN_FOLDER$WORKING$TRAIN$gold_train$char_drs_ext
valid_src=$MAIN_FOLDER$WORKING$DEV$gold_dev$sent_ext$char_sent_ext
valid_tgt=$MAIN_FOLDER$WORKING$DEV$gold_dev$char_drs_ext
log_file=$MAIN_FOLDER$LOG$VOCAB_NAME$log_ext

echo $train_src
echo $train_tgt
echo $valid_src
echo $valid_tgt

$PREPROCESS_LUA -train_src $train_src -train_tgt $train_tgt -valid_src $valid_src -valid_tgt $valid_tgt -save_data $MAIN_FOLDER$VOCAB$VOCAB_NAME -src_words_min_frequency $src_words_min_frequency -tgt_words_min_frequency $tgt_words_min_frequency -src_seq_length $src_seq_length -tgt_seq_length $tgt_seq_length -log_file $log_file $dynamic_dict $src_vocab $tgt_vocab
