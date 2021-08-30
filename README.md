# Multilingual-DRS-Semantic-parsing

This repo contains codes for the following paper: 

*Jingfeng Yang, Federico Fancellu, Bonnie Webber, Diyi Yang*: Frustratingly Simple but Surprisingly Strong: Using Language-Independent Features for Zero-shot Cross-lingual Semantic Parsing. (EMNLP'2021)

If you would like to refer to it, please cite the paper mentioned above. 


## Getting Started
These instructions will get you running the codes.

### Requirements
* Python 3.6 or higher
* Pytorch >= 1.3.0
* OpenNMT
* fairseq

### Data

Before experiments, you should download data from [Parallel Meaning Bank (PMB)](https://pmb.let.rug.nl/data.php). We did experiments on PMB 2.1 and PMB 3.0 in the paper. On PMB 2.1, to fairly compare 6 semnatic parsers including those coarse2fine parsers, output are [tree-structured sequences](https://github.com/EdinburghNLP/EncDecDRSparsing) or [corresponding linearization of DRS with some loss of information](https://github.com/RikVN/Neural_DRS). On PMB 3.0, output are [linearization of DRS without loss of information](https://github.com/RikVN/Neural_DRS).

You also need to use [UDPipe](https://ufal.mff.cuni.cz/udpipe) to get UPOS and UD features.

### Coarse2fine semantic parsing

Coarse2fine semantic parsing code are in coarse2fine-models folder. Detailed running instructions are recorded in [coarse2fine parsing page](https://github.com/JingfengYang/Multilingual-DRS-Semantic-parsing/tree/main/coarse2fine-models).

### LSTM, Transformer and XLM_R encoder semantic parsing with sequential decoding

Code are in N-models folder. Bash scripts in N-models/src/opennmt_scripts will help you run all experiments.

For example, if you want to run LSTM semantic parsing with sequential decoding where Universal Dependency are used as features,
Run
```
bash preprocess-dep.sh &&
bash train.sh &&
bash parse-dep.sh
```

LSTM and Transformer semantic parsing with sequential decoding relied on [OpenNMT](https://github.com/OpenNMT/OpenNMT-py), where DRS parsers are adapted from [Neural DRS](https://github.com/RikVN/Neural_DRS). Since original OpenNMT support word-level extra features, original OpenNMT repo can be reused.

XLM_R encoder semantic parsing with sequential decoding relies on [fairseq](https://github.com/pytorch/fairseq). Because original fairseq did not support extra word-level features, we adapted fairseq code, which is in N-models/fairseq. Also, we adapted it to use different learning rate in encoder and decoder side (because encoder is initialised with pretrained multilingual model while decoder is randomly initialized.) 


## Aknowledgement

Code is adapted from [OpenNMT](https://github.com/OpenNMT/OpenNMT-py), [fairseq](https://github.com/pytorch/fairseq), [Neural DRS](https://github.com/RikVN/Neural_DRS) and [EncDecDRSparsing](https://github.com/EdinburghNLP/EncDecDRSparsing).