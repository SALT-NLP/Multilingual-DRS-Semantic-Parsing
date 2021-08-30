from __future__ import unicode_literals, print_function, division
from io import open
import re

def readfile(filename):
	data = []
	with open(filename,'r',encoding='utf-8') as r:
		while True:
			l1 = r.readline().strip()
			if l1 == "":
				break
			l1List = l1.split()
			l2 = r.readline().strip()
			l2List=l2.split()
			l3 = r.readline().strip()
			l4 = r.readline().strip()
			if l4.endswith('|||'):
				l4 = l4[:-3]
			l4List=l4.split('|||')
			l5 = r.readline().strip()
			l6 = r.readline().strip()
			l6List = l6.split()
			l7 = r.readline().strip()
			l7List=l7.split()
			assert(len(l2List)==len(l1List))
			data.append((l1,l1List,l2,l2List,l3,l4,l4List,l5,l6,l6List,l7,l7List))
	return data

content=readfile('it.input')
with open('it_input.txtRaw', 'w', encoding='utf-8') as writer:
	for l1, l1List, l2, l2List, l3, l4, l4List, l5, l6, l6List, l7, l7List in content:
		deleted=[]
		for token in l4List:
			if token.split('|')[0]=='punct':
				deleted+=[token]
		for token in deleted:
			l4List.remove(token)
		for i,token in enumerate(l6List):
			if '(~' in token:
				term=token.split('(~')
				if term[0].islower():
					if term[0] in l7List:
						print(term)
						range=term[1].split('-')
						assert(len(range)==2)
						word=l1[int(range[0]):int(range[1])]
						print(word)
						if word[0]==' ':
							word = l1[int(range[0])+1:int(range[1])+1]
						if ' ' not in word:
							assert (word in l1List)
							id = l1List.index(word)
							l6List[i] = l2List[id] + '('
						else:
							print(word)
							ws=word.split()
							for w in ws:
								if w not in l1List:
									print('3',token)
							idList=[l1List.index(w) for w in ws]
							print(idList)
							l6List[i]='~'.join([l2List[id] for id in idList])+'('
					elif term[0]=='geological~formation' or term[0]=='phone~number' or term[0]=='musical~organization':
						l6List[i] = term[0].upper() + '('.replace('~','-')
					elif '~' in term[0]:
						#print(term[0])
						#print(l7List)
						for wx in term[0].split('~'):
							if not wx in l7List:
								print(term[0])
								print(l7List)
							if wx in l2List:
								break
							if wx=='ice' or wx=='pacifische' or wx=='verenigd' or wx=='karel' or wx=='roller' \
									or wx=='amerikaanse' or wx =='down' or wx =='officer' or wx=='used' or wx=='cellular' \
									or wx == 'fed' or wx=='eurovisie' or wx=='private' or wx == 'take' or wx=='miniature'\
									or wx =='knitting' or wx=='vereinigte' or wx=='armed':
								break
							assert(wx in l7List)
						range = term[1].split('-')
						assert (len(range) == 2)
						word = l1[int(range[0]):int(range[1])]
						if word[0]==' ':
							word = l1[int(range[0])+1:int(range[1])+1]
						if ' ' not in word:
							assert (word in l1List)
							id = l1List.index(word)
							l6List[i] = l2List[id] + '('
						else:
							ws = word.split()
							idList = [l1List.index(w) for w in ws]
							print('~',idList)
							l6List[i] = '~'.join([l2List[id] for id in idList]) + '('

					else:
						print(token)
						l6List[i] = term[0].upper() + '('
						assert ('~' not in l6List[i])
				else:
					l6List[i]=term[0].upper()+'('
					assert('~' not in l6List[i])
			elif token[-1]=='(':
				print('2',token)
				l6List[i]=token.upper()
			elif token==')':
				pass
			else:
				if not re.match(r'^[A-Z]\d+$',token):
					print(token)
				assert(re.match(r'^[A-Z]\d+$',token))
		writer.write(l1 + '\n')
		writer.write(l2 + '\n')
		writer.write(l3 + '\n')
		writer.write('|||'.join(l4List) + '|||\n')
		writer.write(l5 + '\n')
		writer.write(' '.join(l6List) + '\n')
		writer.write(l7 + '\n')


