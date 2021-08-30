#!/bin/bash
source config/opennmt/default_config.sh #always load default settings from config file here

# Import to set every time, the folder in which we save all experiment results models, working files, output, logs, etc
MAIN_FOLDER="${GIT_HOME}experiments/opennmt/init21-lstm-multi/"
# Gold files - if you do not plan on changing these after you set them,  you can also move them to default_config.sh 
GOLD_FOLDER="/home/jyang690/DRSparsing/SemanticParsingPMB/PMB2/gold/"
GOLD_DE_FOLDER="/home/jyang690/DRSparsing/SemanticParsingPMB/PMB_multi/PMB_de_v2/PMB/gold/"
GOLD_IT_FOLDER="/home/jyang690/DRSparsing/SemanticParsingPMB/PMB_multi/PMB_it_v2/PMB/gold/"
GOLD_NL_FOLDER="/home/jyang690/DRSparsing/SemanticParsingPMB/PMB_multi/PMB_nl_v2/PMB/gold/"

gold_train="train.txt"
gold_dev="dev.txt"
gold_test="test.txt"
gold_de="de.txt"
gold_it="it.txt"
gold_nl="nl.txt"

##### Not necessary to set yourself #####

# Variables that are necessary to set for each experiment
# Again, if you always use a certain setting you can add them in default_config.sh
num_runs=1 				# number of runs, due to the randomness we often like to average over them
var_rewrite="rel" 		# options: {rel, abs, none}
casing="normal"     	# options: {normal, lower, feature}
representation="word"  	# options: {char, word, char_word}
train_steps="15000"			# 15 is good for training with gold data
save_checkpoint_steps="250"
valid_steps="250"
start_decay_steps="3500"             #In "default" decay mode, start decay after this step.
decay_steps="750"
gpuid="6" 						# Use -gpuid 1 if training from GPU, otherwise 0 for cpu
no_sep="--no_sep"
layers="2"
rnn_type="LSTM"
batch_size="12" 		   #default 64
batch_type="sents"
optim="sgd"
learning_rate="0.7"            #Initial learning rate. If adagrad or adam is used, then this is the global learning rate. Recommended settings are: sgd = 1, adagrad = 0.1, adam
max_grad_norm="5"              #Default 5. Clip the gradients L2-norm to this value. Set to 0 to disable.
learning_rate_decay="0.7"      #Default 0.7

src_vocab="-src_vocab /home/jyang690/DRSparsing/SemanticParsingPMB/PMB_multi/src_vocab"
tgt_vocab="-tgt_vocab /home/jyang690/DRSparsing/SemanticParsingPMB/PMB_multi/tgt_vocab"
embed_file="/home/jyang690/DRSparsing/SemanticParsingPMB/PMB_multi/wiki.multi.vec.txt"
fix_word_vec="-fix_word_vecs_enc"

# Obviously, it is also possible to change parameter settings
# You can do that by adding them here, they will override settings in default_config.sh
# Please check that file to know how to change the variables
# In comments are some examples on how to change them

#dropout="0.1"
#learning_rate="0.8"
#beam_size="5"





