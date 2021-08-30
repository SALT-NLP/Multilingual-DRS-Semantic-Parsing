#!/bin/bash
source config/opennmt/default_config.sh #always load default settings from config file here

# Import to set every time, the folder in which we save all experiment results models, working files, output, logs, etc
MAIN_FOLDER="${GIT_HOME}experiments/opennmt/init21/"
# Gold files - if you do not plan on changing these after you set them,  you can also move them to default_config.sh 
GOLD_FOLDER="/home/jyang690/DRSparsing/SemanticParsingPMB/PMB2/gold/"

gold_train="train.txt"
gold_dev="dev.txt"
gold_test="test.txt"

##### Not necessary to set yourself #####

# Variables that are necessary to set for each experiment
# Again, if you always use a certain setting you can add them in default_config.sh
num_runs=1 				# number of runs, due to the randomness we often like to average over them
var_rewrite="rel" 		# options: {rel, abs, none}
casing="normal"     	# options: {normal, lower, feature}
representation="word"  	# options: {char, word, char_word}
train_steps="25600"			# 15 is good for training with gold data
save_checkpoint_steps="40"
valid_steps="40"
start_decay_steps="6000"             #In "default" decay mode, start decay after this step.
decay_steps="1000"
learning_rate_decay="0.9"      #Default 0.7
gpuid="5" 						# Use -gpuid 1 if training from GPU, otherwise 0 for cpu
no_sep="--no_sep"

# Obviously, it is also possible to change parameter settings
# You can do that by adding them here, they will override settings in default_config.sh
# Please check that file to know how to change the variables
# In comments are some examples on how to change them

#dropout="0.1"
#learning_rate="0.8"
#beam_size="5"





