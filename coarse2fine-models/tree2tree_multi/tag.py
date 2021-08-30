# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
import re
from io import open
###global_relation
class Tag:
	def __init__(self, filename, lemmas):
		self.filename = filename

		#......14 8 39 14 13
		self.MAX_KV = 15
		self.MAX_PV = 10
		self.MAX_XV = 40
		self.MAX_EV = 15
		self.MAX_SV = 15
		self.MAX_TV = 9
		self.MAX_T10 = 5
		self.MAX_X10 = 6

		self.SOS = "<SOS>"
		self.EOS = "<EOS>"
		self.reduce = ")"

		self.rel_sdrs = "SDRS("
		self.rel_drs = "DRS("

		self.rel_not = "NOT("
		self.rel_nec = "NEC("
		self.rel_pos = "POS("
		self.rel_or = "OR("
		self.rel_imp = "IMP("

		self.rel_result="RESULTS("
		self.rel_continuation="CONTINUATION("
		self.rel_explanation="EXPLANATION("
		self.rel_contrast="CONTRAST("

		self.rel_clocktime = 'CLOCK_TIME('
		self.rel_cardrel = 'CARD_REL('
		self.rel_eq = "EQ("
		self.word_link='~'

		self.relation_global = list()
		for line in open(filename,'r',encoding='utf-8'):
			line = line.strip()
			if line[0] == "#":
				continue
			self.relation_global.append(line.strip().upper())
		
		self.tag_to_ix = {self.SOS:0, self.EOS:1}
		self.ix_to_tag = [self.SOS, self.EOS]
		
		
		self.tag_to_ix[self.reduce] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.reduce) #2

		self.tag_to_ix[self.rel_sdrs] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_sdrs) #3
		self.tag_to_ix[self.rel_drs] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_drs) #4

		self.tag_to_ix[self.rel_not] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_not) #5
		self.tag_to_ix[self.rel_nec] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_nec) #6
		self.tag_to_ix[self.rel_pos] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_pos) #7
		self.tag_to_ix[self.rel_or] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_or) #8
		self.tag_to_ix[self.rel_imp] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_imp) #9

		self.tag_to_ix[self.rel_result] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_result)  # 10
		self.tag_to_ix[self.rel_continuation] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_continuation)  # 11
		self.tag_to_ix[self.rel_explanation] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_explanation)  # 12
		self.tag_to_ix[self.rel_contrast] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_contrast)  # 13


		self.tag_to_ix[self.rel_eq] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_eq) #14
		self.tag_to_ix[self.rel_clocktime] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_clocktime)#15
		self.tag_to_ix[self.rel_cardrel] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.rel_cardrel)#16

		self.tag_to_ix[self.word_link] = len(self.tag_to_ix)
		self.ix_to_tag.append(self.word_link)  # 17

		self.global_start = len(self.tag_to_ix)
		for tag in self.relation_global:
			if tag in self.tag_to_ix:
				continue
			self.tag_to_ix[tag] = len(self.tag_to_ix)
			self.ix_to_tag.append(tag)

		self.p_rel_start = len(self.tag_to_ix)
		for i in range(self.MAX_PV):
			self.tag_to_ix["P"+str(i+1)+"("] = len(self.tag_to_ix)
			self.ix_to_tag.append("P"+str(i+1)+"(")

		self.p_tag_start = len(self.tag_to_ix)
		for i in range(self.MAX_PV):
			self.tag_to_ix["P"+str(i+1)] = len(self.tag_to_ix)
			self.ix_to_tag.append("P"+str(i+1))
		self.x_tag_start = len(self.tag_to_ix)
		print(self.x_tag_start)
		for i in range(self.MAX_XV):
			self.tag_to_ix["X"+str(i+1)] = len(self.tag_to_ix)
			self.ix_to_tag.append("X"+str(i+1))
		self.e_tag_start = len(self.tag_to_ix)
		for i in range(self.MAX_EV):
			self.tag_to_ix["E"+str(i+1)] = len(self.tag_to_ix)
			self.ix_to_tag.append("E"+str(i+1))
		self.s_tag_start = len(self.tag_to_ix)
		for i in range(self.MAX_SV):
			self.tag_to_ix["S"+str(i+1)] = len(self.tag_to_ix)
			self.ix_to_tag.append("S"+str(i+1))
		self.t_tag_start = len(self.tag_to_ix)
		for i in range(self.MAX_TV):
			self.tag_to_ix["T"+str(i+1)] = len(self.tag_to_ix)
			self.ix_to_tag.append("T"+str(i+1))
		self.t10_tag_start = len(self.tag_to_ix)
		for i in range(self.MAX_T10):
			self.tag_to_ix["Y" + str(i)] = len(self.tag_to_ix)
			self.ix_to_tag.append("Y" + str(i))
		self.x10_tag_start = len(self.tag_to_ix)
		for i in range(self.MAX_X10):
			self.tag_to_ix["Z" + str(i)] = len(self.tag_to_ix)
			self.ix_to_tag.append("Z" + str(i))

		self.tag_size = len(self.tag_to_ix)
		assert len(self.tag_to_ix) == len(self.ix_to_tag)

		self.UNK = "<UNK>("
		self.ix_to_lemma = list()
		for lemma in lemmas:
			assert lemma+'(' not in self.tag_to_ix
			self.tag_to_ix[lemma] = len(self.tag_to_ix)
			self.ix_to_lemma.append(lemma)
		self.all_tag_size = len(self.tag_to_ix)
		with open('lemmaDict.py','w') as w:
			for a,b in self.tag_to_ix.items():
				w.write(a+' '+str(b)+'\n')


	def type(self, string):
		if string in self.ix_to_tag or string.upper() in self.ix_to_tag[:9] :
			return -2, self.tag_to_ix[string.upper()]
		else:
			if string.isupper():
				return -2, self.tag_to_ix['UNK-GLO(']
			return -1, -1
					
