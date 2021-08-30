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

SPM=vcpkg/buildtrees/sentencepiece/src/v0.1.82-78b42c73ce.clean/build/src/spm_encode
MODEL=mbart.cc25/sentence.bpe.model
DATA=$MAIN_FOLDER$WORKING
TRAIN2=$TRAIN$gold_train
VALID2=$DEV$gold_dev
TEST2=$TEST$gold_test
SRC=$sent$char_sent_ext
TGT=$char_drs

${SPM} --model=${MODEL} < $MAIN_FOLDER$WORKING$TRAIN$gold_train$sent_ext$char_sent_ext > ${DATA}${TRAIN2}.spm.${SRC} 
${SPM} --model=${MODEL} < $MAIN_FOLDER$WORKING$TRAIN$gold_train$char_drs_ext > ${DATA}${TRAIN2}.spm.${TGT}
${SPM} --model=${MODEL} < $MAIN_FOLDER$WORKING$DEV$gold_dev$sent_ext$char_sent_ext > ${DATA}${VALID2}.spm.${SRC} 
${SPM} --model=${MODEL} < $MAIN_FOLDER$WORKING$DEV$gold_dev$char_drs_ext > ${DATA}${VALID2}.spm.${TGT} 
${SPM} --model=${MODEL} < $MAIN_FOLDER$WORKING$TEST$gold_test$sent_ext$char_sent_ext > ${DATA}${TEST2}.spm.${SRC} 
${SPM} --model=${MODEL} < $MAIN_FOLDER$WORKING$TEST$gold_test$char_drs_ext > ${DATA}${TEST2}.spm.${TGT} 

DICT=mbart.cc25/dict.txt
fairseq-preprocess \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${DATA}${TRAIN2}.spm \
  --validpref ${DATA}${VALID2}.spm \
  --testpref ${DATA}${TEST2}.spm \
  --destdir ${DATA}$DEST_EN \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --tgtdict ${DICT} \
  --workers 70
