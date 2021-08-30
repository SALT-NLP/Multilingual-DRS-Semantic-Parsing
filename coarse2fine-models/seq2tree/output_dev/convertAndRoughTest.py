from __future__ import unicode_literals, print_function, division
from nltk.tree import *
from io import open
import argparse
import re

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-r1', type=int, default=1)
parser.add_argument('-r2', type=int, default=2)
parser.add_argument('-src', type=str, default='dev.gold')
parser.add_argument('-trg', type=str, default='dev.test')
parser.add_argument('-gold', type=str, default='dev.test')
args = parser.parse_args()

logList=['NOT','NEC','POS','OR','DUPLEX','IMP','RESULTS','CONTINUATION','EXPLANATION','CONTRAST']

def traverseTree(tree,tupleList,refList,kpList,stack,b):
	if tree.label() =='DRS' or tree.label()=='SDRS':
		stack.append(b)
		b=b+1
		node=stack[-1]
		for child in tree:
			assert(isinstance(child, Tree))
			assert not(child.label() =='DRS' or child.label()=='SDRS')
			if re.match(r'^K\d+$',child.label()) or re.match(r'^P\d+$',child.label()):
				List=[]
				List+=['b' + str(node)]
				List+=[child.label()[0]]
				assert (len(child)==1)
				for c in child:
					assert(isinstance(c, Tree))
					if c.label() == 'DRS' or c.label() == 'SDRS':
						List += ['b' + str(b)]
						refList.append(['b' + str(node), child.label(),'b' + str(b)])
						b = traverseTree(c, tupleList,refList,kpList, stack, b)
				kpList+=[List]

			elif child.label().upper() in logList and isinstance(child[0], Tree):
				List = []
				List += ['b' + str(node)]
				List += [child.label()]
				for c in child:
					if not (isinstance(c, Tree)):
						print(c)
						print(child)
					assert (isinstance(c, Tree))
					List += ['b' + str(b)]
					if c.label() == 'DRS' or c.label() == 'SDRS':
						b = traverseTree(c, tupleList,refList,kpList, stack, b)
				tupleList += [List]

			else:
				List = []
				List += ['b' + str(node)]
				List += [child.label()]
				for c in child:
					assert not (isinstance(c, Tree))
					List += [c]
				tupleList += [List]
		stack.pop()
	return b

def rename(tupleList,kpList,refList):
	dict={}
	for ref in refList:
		if not len(ref) == 3:
			print(refList)
		assert (len(ref) == 3)

	for _ , ref , drs in kpList:
		if ref in dict:
			print(kpList)
		assert not (ref in dict)
		dict[ref]=drs

	for tuple in tupleList:
		assert (len(tuple) == 3 or len(tuple)==4)
		if tuple[2] in dict:
			tuple[2]=dict[tuple[2]]
		if len(tuple)==4:
			if tuple[3] in dict:
				tuple[3]=dict[tuple[3]]

def normalize(tupleList):
	for tuple in tupleList:
		assert(re.match(r'^(S|X|E|T|Y|Z|b)\d+$',tuple[2]))
		tuple[2]=tuple[2].lower()
		if len(tuple)==4:
			if not re.match(r'^((S|X|E|T|Y|Z|b)\d+)|CARD_NUMBER|TIME_NUMBER$', tuple[3]):
				print(tuple)
			assert (re.match(r'^((S|X|E|T|Y|Z|b)\d+)|CARD_NUMBER|TIME_NUMBER$', tuple[3]))
			if  tuple[3]=='TIME_NUMBER' or tuple[3]=='CARD_NUMBER':
				tuple[3]='"'+tuple[3]+'"'
			else:
				tuple[3]=tuple[3].lower()

def main():
	reader = open(args.src, 'r', encoding='utf-8')
	writer = open(args.trg, 'w', encoding='utf-8')
	for line in reader:
		tokens = line.strip().split()
		line = ' '.join(['(' + token[:-1] if token[-1] == '(' else token for token in tokens])
		tree = Tree.fromstring(line)
		tupleList = []
		refList = []
		kpList = []
		stack = []
		b = 0
		traverseTree(tree, tupleList, refList, kpList, stack, b)
		rename(tupleList, refList, kpList)
		normalize(tupleList)
		for tuple in tupleList:
			writer.write(' '.join(tuple) + '\n')
		writer.write('\n')
	reader.close()
	writer.close()


	for i in range(args.r1, args.r2):
		reader = open(str(i) + '.drs', 'r', encoding='utf-8')
		writer = open(str(i) + '.test', 'w', encoding='utf-8')
		for line in reader:
			tokens = line.strip().split()
			line = ' '.join(['(' + token[:-1] if token[-1] == '(' else token for token in tokens])
			tree = Tree.fromstring(line)
			tupleList = []
			refList = []
			kpList = []
			stack = []
			b = 0
			traverseTree(tree, tupleList, refList, kpList, stack, b)
			rename(tupleList, refList, kpList)
			normalize(tupleList)
			for tuple in tupleList:
				writer.write(' '.join(tuple) + '\n')
			writer.write('\n')
		reader.close()
		writer.close()

	maxF=0
	maxFindex=1
	for idx in range(args.r1, args.r2):
		readerTest = open(str(idx) + '.test', 'r', encoding='utf-8')
		readerGold = open(args.gold, 'r', encoding='utf-8')
		linesTest = []
		linesGold = []
		sentence = []
		for line in readerTest:
			if line.strip() == '':
				linesTest.append(sentence)
				sentence = []
			else:
				sentence.append(line.strip())
		sentence = []
		for line in readerGold:
			if line.strip() == '':
				linesGold.append(sentence)
				sentence = []
			else:
				sentence.append(line.strip())
		readerTest.close()
		readerGold.close()

		totalGold = 0
		totalTest = 0
		right = 0
		if not len(linesTest) == len(linesGold):
			print(len(linesTest), len(linesGold))
		assert (len(linesTest) == len(linesGold))
		for i, (test, gold) in enumerate(zip(linesTest, linesGold)):
			testSet = set(test)
			goldSet = set(gold)
			rightSet = testSet.intersection(goldSet)
			totalGold += len(goldSet)
			totalTest += len(testSet)
			right += len(rightSet)

		precision = right / totalTest
		recall = right / totalGold
		f=2 * precision * recall / (precision + recall)
		print(str(idx) + ': p %.10f, r: %.10f, f: %.10f ' % (precision, recall, f))
		if f>maxF:
			maxF=f
			maxFindex=idx
	print('max: %d, f: %.10f' % (maxFindex,maxF))


if __name__ == "__main__":
	main()
