#!/bin/bash
# Default settings that are imported for each experiment
# Might be overwritten per individual experiment in different config files

## First we list the settings that have to be present in each config file and never here!
## That is the reason they are commented

# MAIN_FOLDER=""
# GOLD_FOLDER=""
# gold_train=""
# gold_dev=""
# var_rewrite=""
# representation=""
# end_epoch=""
# num_runs=1

###### IMPORTANT TO SET THESE ######

GIT_HOME=""	# where the Git Neural_DRS repository is located
ONMT_HOME="OpenNMT-py/"		# where OpenNMT is located
sig_file="${GIT_HOME}DRS_parsing/evaluation/clf_signature.yaml" # signature file for clf_referee, check if correct location

# File from which to train pre-trained embeddings. When recreating our experiments, you can download these embeddings here:
# http://www.let.rug.nl/rikvannoord/DRS/embeddings/
# Not necessary to set this parameter if you only do character-level training
embed_file="/home/jyang690/DRSparsing/SemanticParsingPMB/PMB_multi/wiki.multi.en.vec.txt"
embed_type="word2vec"

##### PARAMETER SETTINGS FOR EXPERIMENTS #####

# These are the default settings that will be used if you do not specify them in your own config file
# If you do specify them in your own config file, these values will be overridden

# Parameter settings for preprocessing
src_words_min_frequency="3"
tgt_words_min_frequency="3"
src_seq_length="1000"
tgt_seq_length="1000"

# Parameter settings for training
src_word_vec_size="300"
tgt_word_vec_size="300"
layers="6"
rnn_size="300"
model_type="transformer" #options LSTM (default), GRU
transformer_ff="-transformer_ff 2048"
heads="-heads 6"
pos_enc="-position_encoding"
transformer_init="-param_init 0  -param_init_glorot"
dropout="0.2"
dropout_input=""           #boolean, use as -dropout_input (default false)
dropout_words="0"          #dropout probability applied to the source sequence (default 0)
dropout_type="naive"       #dropout type, options naive (default) or variational
residual=""           	   #boolean, add residual connections between recurrent layers (default empty is false)
bridge="-bridge"         	   #define how to pass encoder states to the decoder. With copy, the encoder and decoder must have the same number of layers. Accepted: copy, dense, dense_nonlinear, none; default: copy
encoder_type="brnn"        #accepted: rnn, brnn, dbrnn, pdbrnn, gnmt, cnn; default: rnn
attention="global"         #none or global (default)
max_pos="1000"             #maximum value for positional indexes (default 50)
global_attention="general" #accepted: general, dot, concat; default: general
copy_att="-copy_attn -copy_attn_type dot"
dynamic_dict="-dynamic_dict"

#Trainer/optimizer options
report_every="10"             #default 50
validation_metric="perplexity" #accepted: perplexity, loss, bleu, ter, dlratio; default: perplexity
batch_size="512" 		   #default 64
batch_type="tokens"
optim="adam" 				   #optimizer, accepted: sgd, adagrad, adadelta, adam
learning_rate="0.001"            #Initial learning rate. If adagrad or adam is used, then this is the global learning rate. Recommended settings are: sgd = 1, adagrad = 0.1, adam = 0.0002.
max_grad_norm="5"              #Default 5. Clip the gradients L2-norm to this value. Set to 0 to disable.
learning_rate_decay="0.7"      #Default 0.7
train_from=""				   #Add this if we want to train from a checkpoint (use -train_from FOLDER and -continue as well)

# Parameter settings for testing
batch_size_test="12"        #batch size test
beam_size="10"
max_sent_length="1000"		#default 250
replace_unk="-replace_unk" 	#boolean, default empty
n_best="1" 					#If > 1, it will also output an n-best list of decoded sentences.

length_norm="-length_penalty wu -alpha 0.90"
coverage_norm=''
#coverage_norm="-coverage_penalty wu -beta 0.0"			#Coverage normalization coefficient (beta). An extra coverage term multiplied by beta is added to hypotheses scores. If is set to 0 (default), no coverage normalization.
log_level="WARNING" 		#accepted: DEBUG, INFO, WARNING, ERROR, NONE; default: INFO

# Create script names already here 

SRCPATH="${GIT_HOME}src/"

ens_py="ensemble_best_models.py"
cv_py="create_cv_files.py"
stat_py="create_stat_files.py"
avg_exp_score="avg_exp_score.py"

PREPROCESS_PYTHON="${SRCPATH}preprocess2.py"
PREPROCESS_LUA="onmt_preprocess"
PREPROCESS_SH="${SRCPATH}/opennmt_scripts/preprocess.sh"
EMBED_LUA="python3 ${ONMT_HOME}tools/embeddings_to_torch.py"
TRAIN_LUA="onmt_train"
TRAIN_SH="${SRCPATH}/opennmt_scripts/train.sh"
PARSE_SH="${SRCPATH}opennmt_scripts/parse.sh"
TRANSLATE_LUA="onmt_translate"
POSTPROCESS_PY="${SRCPATH}postprocess2.py"

# Embedding names and files

embed="embeddings/embedding"
src_id="src"
tgt_id="tgt"
emb_ext="-embeddings-300.t7"
pre_word_vecs_enc=""
pre_word_vecs_dec=""

# Extensions of files we will create
char_drs_ext=".char.drs"
char_drs="char.drs"
char_sent_ext=".char.sent"
char_sent="char.sent"
sent_ext=".raw"
sent="raw"
lex_ext=".lex"
valid_drs_ext=".valid.drs"
var_drs_ext=".var"
output_ext=".seq.drs"
log_ext=".log"
res_ext=".res"
eval_ext=".txt"
MODEL_EXT="*.t7"
txt_ext=".txt"
feat_vec=""

# Log file
log_file_train="train.log"

# Names of things we add to file
VOCAB_NAME="vcb"

# Vocab we train from (usually $MAIN_FOLDER$VOCAB$VOCAB_NAME$ext_src except for pretrain experiments)

ext_voc=".vocab.pt"

# List of folder names

WORKING="working/"
VOCAB="vocab/"
MODELS="models/"
OUTPUT="output/"
EVAL="evaluation/"
LOG="log/"
TRAIN="train/"
DEV="dev/"
TEST="test/"
DE="de/"
IT="it/"
NL="nl/"
DEST_EN="dest-en"
DEST_DE="dest-de"
DEST_IT="dest-it"
DEST_NL="dest-nl"
