# SemanticParsingPMB

## Mono-lingual Semantic parsing

### Preparation
1. Copy `sskip.100.vectors` to PMB2/gold
2. Copy `train_input.txtRaw` to PMB2/gold
3. Copy `dev_input.txtRaw` to PMB2/gold
4. Copy `test_input.txtRaw` to PMB2/gold

### Pre-processing
1. 
>cd PMB2/gold
2. 
>python replaceCard.py

>python replaceCard.py -src dev_input.txtRaw -trg dev_input.txt

>python replaceCard.py -src test_input.txtRaw -trg test_input.txt
3. Generate the global condition tag file `tag.txt` from the training data
>python globalRel.py

### Run
1. 
>`cd seq2tree` (or `cd tree2tree`  or `cd tree2treePos` or `cd seqtree2tree`)
2. 
>python main.py
```
	-s  if the model will be saved?  default=False
	-t  if only test the saved model?  default=False
	-r  if reload the model when continuing to train?  default=False
	-mp  if delete the Universal POS tags as features?  default=False
	-md  if delete the Universal Dependency tags as features?  default=False
	-mw  if delete the word embeddings as features?  default=False
	-model the model name   default='output_model/1.model'
```
3. use Ctrl+c to stop the training

### Evaluation
The devlopment and test results after each epoch are listed in output_dev and output_tst respectively. Firstly, choose the epoch which performs best on the devlopment set. Then get the test results on the test set. 

Evaluate on the development set.

1. 
>`cd seq2tree/output_dev` (or `cd tree2tree/output_dev`  or `cd tree2treePos/output_dev` or `cd seqtree2tree/output_dev`)
2. Transform the formation of the result Discourse Representation Structure to lines which fit to Counter, and generate the rough scores by comparing each line of the two files. Choose the epoch (file) i with the highest f-score as the i of the next step.
>python convertAndRoughTest.py 
```
	-r1 startpoint of the tested file range (epoch number)  default=1
	-r2  endpoint of the tested file range (epoch number)  default=2
	-src  the gold development file   default='dev.gold'
	-trg  transformation results of the gold development file  default='dev.test'
	-gold  the gold development file after transformation  default='dev.test'
```
3. 
> python ../../DRS_parsing/counter.py -f1 i.test -f2 dev.test -pr -prin -ms (> i.results)

Evaluate on the test set. (Roughly the same as the evaluation on the development set. For example, `cd seq2tree/output_tst`)

### Error analysis
1. 
>jupyter notebook
2. Open analysis_dmatch.ipynb
3. Change the file name in the 3rd and 4th cell an run the first 4 cells

## Multi-lingual Semantic parsing

### Preparation
1. Copy `wiki.multi.de.vec.txt` to PMB_multi
2. Copy `wiki.multi.it.vec.txt` to PMB_multi
3. Copy `wiki.multi.nl.vec.txt` to PMB_multi
4. Copy `wiki.multi.en.vec.txt` to PMB_multi
5. Copy `de.input` to PMB_multi/PMB_de_v2/PMB/gold
6. Copy `nl.input` to PMB_multi/PMB_nl_v2/PMB/gold
7. Copy `it.input` to PMB_multi/PMB_it_v2/PMB/gold


### Pre-processing
1. 
>cd PMB2/gold
2. 
>python replaceCard.py

>python replaceCard.py -src dev_input.txtRaw -trg dev_input.txt
3. Generate the global condition tag file `tag.txt` from the training data
>python globalRel.py
4. 
>cd PMB_multi/PMB_de_v2/PMB/gold
5. 
>python replace1.py
6. 
>python replace2.py
7. 
Replace step 4 , 5 and 6 on nl and it.

### Run
1. 
>`cd seq2tree_multi` (or `cd tree2tree_multi`  or `cd tree2treePos_multi` or `cd seqtree2tree_multi`)
2. 
>python main.py
```
	-s  if the model will be saved?  default=False
	-t  if only test the saved model?  default=False
	-r  if reload the model when continuing to train?  default=False
	-mp  if delete the Universal POS tags as features?  default=False
	-md  if delete the Universal Dependency tags as features?  default=False
	-mw  if delete the cross-lingual word embeddings as features?  default=False
	-model the model name   default='output_model/1.model'
```
3. use Ctrl+c to stop the training

### Evaluation
The devlopment and test results after each epoch are listed in output_dev and output_tst respectively. Firstly, choose the epoch which performs best on the devlopment set. Then get the test results on the test set. 

Evaluate on the development set.

1. 
>`cd seq2tree_multi/output_dev` (or `cd tree2tree_multi/output_dev`  or `cd tree2treePos_multi/output_dev` or `cd seqtree2tree_multi/output_dev`)
2. Transform the formation of the result Discourse Representation Structure to lines which fit to Counter, and generate the rough scores by comparing each line of the two files. Choose the epoch (file) i with the highest f-score as the i of the next step.
>python convertAndRoughTest.py 
```
	-r1 startpoint of the tested file range (epoch number)  default=1
	-r2  endpoint of the tested file range (epoch number)  default=2
	-src  the gold development file   default='dev.gold'
	-trg  transformation results of the gold development file  default='dev.test'
	-gold  the gold development file after transformation  default='dev.test'
```
3. 
> python ../../DRS_parsing/counter.py -f1 i.test -f2 dev.test -pr -prin -ms (> i.results)

Evaluate on the nl/de/it test set. (Roughly the same as the evaluation on the development set. For example, `cd seq2tree_multi/output_it_tst`)

### Error analysis
1. 
>jupyter notebook
2. Open analysis_dmatch.ipynb
3. Change the file name in the 3rd and 4th cell an run the first 4 cells

## Acknowledgments
The code is based on the code in [EdinburghNLP/EncDecDRSparsing](https://github.com/EdinburghNLP/EncDecDRSparsing).
