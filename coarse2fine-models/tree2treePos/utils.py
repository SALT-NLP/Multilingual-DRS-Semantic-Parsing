# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import torch
from tree import Tree

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def readfile(filename):
	data = []
	with open(filename,'r',encoding='utf-8') as r:
		while True:
			l1 = r.readline().strip()
			if l1 == "":
				break
			l2 = r.readline().strip()
			l3 = r.readline().strip()
			raw=r.readline().strip()
			if raw.endswith('|||'):
				raw=raw[:-3]
			rels=[]
			heads=[]
			deps=[]
			l4 = [term.split('|') for term in raw.split('|||') if
				  not (term.split('|')[0] == 'nsubj:xsubj' or term.split('|')[0] == 'nsubjpass:xsubj' )]
			for term in raw.split('|||'):
				tokens=term.split('|')
				rels.append(tokens[0])
				heads.append(tokens[1])
				deps.append(tokens[2])
			delete=[]
			for i,rel in enumerate(rels):
				if rel=='conj:and' or rel=='conj:or' or rel=='conj:but':
					r1=heads[i]
					r2=deps[i]
					listr1=[]
					for j, head in enumerate(heads):
						if head==r1:
							listr1+=[deps[j]]
					for j, head in enumerate(heads):
						if head==r2 and deps[j] in listr1:
							delete+=[j]
					for j, dep in enumerate(deps):
						if dep==r2 and not i==j:
							delete += [j]
			l4=[term for i,term in enumerate(l4) if not i in delete]
			l5 = r.readline().strip()
			l6 = r.readline().strip()
			l7 = r.readline()

			data.append((l1.split(), [w.lower() for w in l1.split()], l2.split(), l3.split(),
						 l4, l5.split(), l6.split()))
	return data

def readpretrain(filename):
	data = []
	with open(filename,'r',encoding='utf-8') as r:
		while True:
			l = r.readline().strip()
			if l == "":
				break
			data.append(l.split())
	return data

def computeMaxLength(trn_instances,dev_instances,tst_instances):
	maxLength=0
	for instance in trn_instances:
		if instance[0].size()[0]>maxLength:
			maxLength=instance[0].size()[0]
	for instance in dev_instances:
		if instance[0].size()[0]>maxLength:
			maxLength=instance[0].size()[0]
	for instance in tst_instances:
		if instance[0].size()[0]>maxLength:
			maxLength=instance[0].size()[0]
	return maxLength


def get_from_ix(w, to_ix, unk):
	if w in to_ix:
		return to_ix[w]

	assert unk != -1, "no unk supported"
	return unk

def data2instance(trn_data, ixes,outPutFile):
	instances = []
	deleted=0
	writer = open(outPutFile, 'w', encoding='utf-8')
	for one in trn_data:
		flag=0
		use=[]
		for terms in one[4]:
			use+=[int(terms[2])]
		instances.append([])
		instances[-1].append(torch.tensor([get_from_ix(w, ixes[0][0], ixes[0][1]) for i,w in enumerate(one[0]) if i in use],dtype=torch.long, device=device))
		instances[-1].append(torch.tensor([get_from_ix(w, ixes[1][0], ixes[1][1]) for i,w in enumerate(one[1]) if i in use],dtype=torch.long, device=device))

		instances[-1].append(torch.tensor([get_from_ix(w, ixes[2][0], ixes[2][1]) for i,w in enumerate(one[2]) if i in use],dtype=torch.long, device=device))
		tempOne2=[w for i,w in enumerate(one[2]) if i in use]

		instances[-1].append([])
		relation = 0
		for item in one[6]:
			type, idx = ixes[3].type(item)
			if type == -2:
				assert not (idx == 17)
				if idx == 2:
					if relation == 0:
						instances[-1][-1].append(idx)
					elif relation > 0:
						relation -= 1
					else:
						assert False
				elif idx >= 3 and idx <= 13:
					instances[-1][-1].append(idx)
				elif idx >= ixes[3].p_rel_start and idx < ixes[3].p_tag_start:
					instances[-1][-1].append(idx)
				elif idx >= 14 and idx < ixes[3].p_rel_start:
					relation += 1
			else:
				relation += 1

		instances[-1].append([])
		stack = []
		pointer = 0
		for item in one[6]:
			type, idx = ixes[3].type(item)
			assert not (idx == 17)
			if type == -2:
				if idx == 2:
					if isinstance(stack[-1],list):
						for token in stack[-1]:
							assert not (isinstance(stack[-2],list))
							instances[-1][-1][stack[-2][1]].append([token[1],token[0]])
					elif stack[-1][0] >= 5 and stack[-1][0] <= 13:
						pass
					elif stack[-1][0] >= ixes[3].p_rel_start and stack[-1][0] < ixes[3].p_tag_start:
						pass
					elif stack[-1][0] == 3 or stack[-1][0] == 4:
						pass
					else:
						instances[-1][-1][stack[-2][1]].append([stack[-1][1], stack[-1][0]])
					stack.pop()
				elif idx == 4:
					stack.append((idx, pointer))
					instances[-1][-1].append([])
					pointer += 1
				elif idx == 3 or (idx >= 5 and idx <= 16) or (
						idx >= ixes[3].global_start and idx < ixes[3].p_tag_start):
					stack.append((idx, -2, -1))
			else:
				if '~' in item[:-1]:
					flag2 = 0
					for iy in item[:-1].split('~'):
						if not iy in tempOne2:
							print('item:', (len(instances) + deleted) * 7, item[:-1], iy)
							instances = instances[:-1]
							deleted += 1
							flag = 1
							flag2 = 1
							break
					if flag2 == 1:
						break
				else:
					if not item[:-1] in tempOne2:
						print('item:', (len(instances) + deleted) * 7, item[:-1])
						instances = instances[:-1]
						deleted += 1
						flag = 1
						break
				if '~' in item[:-1]:
					listS=[]
					tokens=item[:-1].split('~')
					for token in tokens:
						type = tempOne2.index(token)
						idx = instances[-1][2].tolist()[type] + ixes[3].tag_size
						assert type != -1 and idx != -1, "unrecogized local relation"
						listS.append((idx, type, -1))
						listS.append((17, -2, -1))
					listS.pop()
					stack.append(listS)
				else:
					type = tempOne2.index(item[:-1])
					idx = instances[-1][2].tolist()[type] + ixes[3].tag_size
					assert type != -1 and idx != -1, "unrecogized local relation"
					stack.append((idx, type, -1))
		if flag==1:
			continue

		instances[-1].append([])
		for item in one[6]:
			type, idx = ixes[3].type(item)
			assert not (idx == 17)
			if type == -2 and idx >= ixes[3].p_tag_start and idx < ixes[3].tag_size:  # variable
				pass
			else:
				if type == -2:
					instances[-1][-1].append(idx)
				else:
					if '~' in item[:-1]:
						tokens = item[:-1].split('~')
						for token in tokens:
							type = tempOne2.index(token)
							idx = instances[-1][2].tolist()[type] + ixes[3].tag_size
							assert type != -1 and idx != -1, "unrecogized local relation"
							instances[-1][-1].append(idx)
							instances[-1][-1].append(17)
						instances[-1][-1].pop()
					else:
						type = tempOne2.index(item[:-1])
						idx = instances[-1][2].tolist()[type] + ixes[3].tag_size

						assert type != -1 and idx != -1, "unrecogized local relation"
						instances[-1][-1].append(idx)

		assert len(instances[-1][-1]) != 0

		instances[-1].append([])
		for item in one[6]:
			type, idx = ixes[3].type(item)
			assert not (idx == 17)
			if type == -2 and idx >= ixes[3].p_tag_start and idx < ixes[3].tag_size:  # variable
				instances[-1][-1][-1].append(idx)
			elif type == -1 or (type == -2 and idx >= 14 and idx < ixes[3].p_rel_start):
				if len(instances[-1][-1]) != 0:
					if not (len(instances[-1][-1][-1]) <= 2 and len(instances[-1][-1][-1]) > 0):
						print(one[6], item)
						print(instances[-1][-1])
					assert len(instances[-1][-1][-1]) <= 2 and len(instances[-1][-1][-1]) > 0
				instances[-1][-1].append([])

		temprate=[]
		for i in range(len(one[0])):
			temprate.append(Tree(i))

		for terms in one[4]:
			temprate[int(terms[2])].tag=get_from_ix(terms[0], ixes[4][0], ixes[4][1])
			if not int(terms[1])==-1:
				temprate[int(terms[1])].add_child(temprate[int(terms[2])])
			if terms[0] =='root':
				root=temprate[int(terms[2])]
		dpList=[w for i,w in enumerate(temprate) if i in use]
		instances[-1].append(dpList)
		instances[-1].append(root)
		assert(len(instances[-1][-2])==instances[-1][0].size(0))
		number=0
		idList=[]
		number,idList,tag=traverseTree(root,number,idList)

		if tag==0:
			for tree in instances[-1][-2]:
				print('f',tree.idx)
				for child in tree.children:
					print('d',child.idx)
			print('one',one[4])
			print('rfalsetree:', (len(instances) + deleted) * 7)
		assert (tag == 1)
		if not number==len(instances[-1][-2]):
			print('falsetree:',(len(instances)+deleted)*7)
			print('use',use)
			instances=instances[:-1]
			deleted+=1
			continue
		assert(number==len(instances[-1][-2]))
		assert(set(idList)==set(use))
		for i, tree in enumerate(instances[-1][-2]):
			tree.idx=i
		instances[-1].append(torch.tensor([get_from_ix(w, ixes[5][0], ixes[5][1])
										   for i, w in enumerate(one[3]) if i in use], dtype=torch.long, device=device))
		instances[-1].append([w for i, w in enumerate(one[2]) if i in use])
		writer.write(' '.join(one[6])+'\n')
	writer.close()
	print(deleted,len(instances)+deleted)
	return instances

def traverseTree(tree,number,idlist):
	number+=1
	idlist+=[tree.idx]
	if len(tree.children)>0:
		for child in tree.children:
			if not child.idx in idlist:
				number, idlist, tag = traverseTree(child, number, idlist)
				if tag==0:
					print(idlist)
					return number, idlist, tag
			else:

				return number, idlist, 0
	return number,idlist,1

