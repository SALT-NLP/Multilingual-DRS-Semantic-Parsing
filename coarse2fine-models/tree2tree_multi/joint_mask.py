from __future__ import unicode_literals, print_function, division
class OuterMask:
	def __init__(self, tags_info):
		self.tags_info = tags_info
		self.mask = 0
		self.need = 1

		self.SOS = tags_info.tag_size*10
		self.relation = tags_info.tag_size*10+1

		self.variable_offset = 0
		self.relation_offset = 1
		self.p_relation_offset = 2
		self.drs_offset = 3
		self.six_offset = 4
		self.four_offset = 5
		
		self.reset()
	def reset(self, ):
		self.relation_count = 0
		self.stack = [self.SOS]
		self.stack_ex = [[0 for i in range(6)]]
		self.stack_variables = []
		self.p = 1

	def get_step_mask(self):
		if self.stack[-1] == self.SOS:
			#SOS
			return self._get_sos_mask()
		elif self.stack[-1] == 3:
			#SDRS
			return self._get_sdrs_mask()
		elif self.stack[-1] == 4:
			#DRS
			return self._get_drs_mask()
		elif self.stack[-1] in [5,6,7]:
			#not, nec, pos
			return self._get_1_mask()
		elif self.stack[-1] in [8,9]:
			#or, imp, duplex
			return self._get_2_mask()
		elif self.stack[-1] in [10,11,12,13]:
			#results continuation explanation contrast
			return self._get_2_mask()
		elif self.stack[-1] == self.tags_info.p_rel_start:
			#p
			return self._get_1_mask()
		else:
			assert False
	def _get_sos_mask(self):
		if self.stack_ex[-1][self.drs_offset] == 0:
			re = self._get_zeros(self.tags_info.tag_size)
			re[self.tags_info.tag_to_ix[self.tags_info.rel_drs]] = self.need
			re[self.tags_info.tag_to_ix[self.tags_info.rel_sdrs]] = self.need
			return re
		else:
			re = self._get_zeros(self.tags_info.tag_size)
			re[1] = self.need
			return re
	def _get_sdrs_mask(self):
		#SDRS
		if self.stack_ex[-1][self.four_offset] < 1:
			re = self._get_zeros(self.tags_info.tag_size)
			re[10] = self.need
			re[11] = self.need
			re[12] = self.need
			re[13] = self.need

			return re
		else:
			#only reduce
			re = self._get_zeros(self.tags_info.tag_size)
			re[self.tags_info.tag_to_ix[self.tags_info.reduce]] = self.need
			return re
	def _get_drs_mask(self):
		re = self._get_zeros(self.tags_info.tag_size)
		re[self.tags_info.tag_to_ix[self.tags_info.reduce]] = self.need
		if self.relation_count <= 40:
			if self.p <= self.tags_info.MAX_PV:
				idx = self.tags_info.p_rel_start + self.p - 1
				re[idx] = self.need
			re[5] = self.need
			re[6] = self.need
			re[7] = self.need
			re[8] = self.need
			re[9] = self.need
		return re

	def _get_1_mask(self):
		if self.stack_ex[-1][self.drs_offset] == 0:
			re = self._get_zeros(self.tags_info.tag_size)
			if self.relation_count <= 40:
				re[3] = self.need
			re[4] = self.need
			return re
		else:
			re = self._get_zeros(self.tags_info.tag_size)
			re[2] = self.need
			return re
	def _get_2_mask(self):
		if self.stack_ex[-1][self.drs_offset] <= 1:
			re = self._get_zeros(self.tags_info.tag_size)
			if self.relation_count <= 40:
				re[3] = self.need
			re[4] = self.need
			return re
		else:
			re = self._get_zeros(self.tags_info.tag_size)
			re[2] = self.need
			return re

	def update(self, type, ix):
		if ix >= 3 and ix <= 13:
			self.stack.append(ix)
			self.relation_count += 1
			self.stack_ex.append([0 for i in range(6)])
		elif ix >= self.tags_info.p_rel_start and ix < self.tags_info.p_tag_start:
			self.stack.append(self.tags_info.p_rel_start)
			self.relation_count += 1
			self.stack_ex.append([0 for i in range(6)])
			self.p += 1
		elif ix == 2:
			self.stack_ex.pop()
			if self.stack[-1] == 3 or self.stack[-1] == 4:
				self.stack_ex[-1][self.drs_offset] += 1
			elif self.stack[-1] >= 5 and self.stack[-1] <= 9:
				self.stack_ex[-1][self.six_offset] += 1
			elif self.stack[-1] >= 10 and self.stack[-1] <= 13:
				self.stack_ex[-1][self.four_offset] += 1
			elif self.stack[-1] == self.tags_info.p_rel_start:
				self.stack_ex[-1][self.p_relation_offset] += 1
			else:
				assert False
			self.stack.pop()
		elif ix == 1:
			pass
		else:
			assert False

	def _get_zeros(self, size):
		return [self.mask for i in range(size)]

	def _get_ones(self, size):
		return [self.need for i in range(size)]

class RelationMask:
	def __init__(self, tags_info, encoder_input_size=0):
		self.tags_info = tags_info
		self.mask = 0
		self.need = 1
		self.SOS=-1
		self.reset(encoder_input_size)
	def reset(self, encoder_input_size):
		self.encoder_input_size = encoder_input_size
		self.is_sdrs = False
		self.prev_word=self.SOS
		self.link=0

	def set_sdrs(self, is_sdrs):
		self.is_sdrs = is_sdrs

	def _get_relations(self):
		if self.link==1:
			assert(self.prev_word>=self.tags_info.tag_size)
			res = self._get_zeros(self.tags_info.tag_size) + self._get_ones(self.encoder_input_size)
			if not self.encoder_input_size == 1:
				res[self.prev_word]=self.mask
		else:
			res = self._get_zeros(self.tags_info.tag_size) + self._get_ones(self.encoder_input_size)
			res[1] = self.need
			idx = 14
			while idx < self.tags_info.p_rel_start:
				res[idx] = self.need
				idx += 1
			if self.prev_word<self.tags_info.tag_size:
				res[17]=self.mask
		return res

	def get_step_mask(self, least,word):
		if word==17:
			self.link=1
		else:
			self.prev_word = word
			self.link=0
		relations = self._get_relations()
		if least:
			relations[1] = self.mask

		return relations

	
	def _get_zeros(self, size):
		return [self.mask for i in range(size)]

	def _get_ones(self, size):
		return [self.need for i in range(size)]

class VariableMask:
	def __init__(self, tags_info):
		self.tags_info = tags_info
		self.mask = 0
		self.need = 1

		self.reset(0)

	def reset(self, p_max):
		self.p_max = p_max
		self.x = 1
		self.e = 1
		self.s = 1
		self.t=1
		self.t10=1
		self.x10=1

		self.stack = []
		self.stack_drs = []

		self.prev_variable = -1
		self.pre_prev_variable = -1

	def get_step_mask(self):
		if self.stack_drs[-1] == 5:
			return self._get_sdrs_mask()
		else:
			return self._get_drs_mask()

	def _get_drs_mask(self):
		if self.prev_variable == -1:
			re = self._get_zeros(self.tags_info.tag_size)

			idx = self.tags_info.x_tag_start
			while idx < self.tags_info.e_tag_start and idx < self.tags_info.x_tag_start + self.x:
				re[idx] = self.need
				idx += 1

			idx = self.tags_info.e_tag_start
			while idx < self.tags_info.s_tag_start and idx < self.tags_info.e_tag_start + self.e:
				re[idx] = self.need
				idx += 1

			idx = self.tags_info.s_tag_start
			while idx < self.tags_info.t_tag_start and idx < self.tags_info.s_tag_start + self.s:
				re[idx] = self.need
				idx += 1

			idx = self.tags_info.t_tag_start
			while idx < self.tags_info.t10_tag_start and idx < self.tags_info.t_tag_start + self.t:
				re[idx] = self.need
				idx += 1

			idx = self.tags_info.t10_tag_start
			while idx < self.tags_info.x10_tag_start and idx < self.tags_info.t10_tag_start + self.t10:
				re[idx] = self.need
				idx += 1

			idx = self.tags_info.x10_tag_start
			while idx < self.tags_info.tag_size and idx < self.tags_info.x10_tag_start + self.x10:
				re[idx] = self.need
				idx += 1

			idx = self.tags_info.p_tag_start
			while idx < self.tags_info.x_tag_start and idx < self.tags_info.p_tag_start + self.p_max:
				re[idx] = self.need
				idx += 1

			return re
		elif self.prev_prev_variable == -1:
			re = self._get_zeros(self.tags_info.tag_size)
			re[1] = self.need

			idx = self.tags_info.x_tag_start
			while idx < self.tags_info.e_tag_start and idx < self.tags_info.x_tag_start + self.x:
				re[idx] = self.need
				idx += 1
			idx = self.tags_info.e_tag_start
			while idx < self.tags_info.s_tag_start and idx < self.tags_info.e_tag_start + self.e:
				re[idx] = self.need
				idx += 1

			idx = self.tags_info.s_tag_start
			while idx < self.tags_info.t_tag_start and idx < self.tags_info.s_tag_start + self.s:
				re[idx] = self.need
				idx += 1

			idx = self.tags_info.t_tag_start
			while idx < self.tags_info.t10_tag_start and idx < self.tags_info.t_tag_start + self.t:
				re[idx] = self.need
				idx += 1

			idx = self.tags_info.t10_tag_start
			while idx < self.tags_info.x10_tag_start and idx < self.tags_info.t10_tag_start + self.t10:
				re[idx] = self.need
				idx += 1

			idx = self.tags_info.x10_tag_start
			while idx < self.tags_info.tag_size and idx < self.tags_info.x10_tag_start + self.x10:
				re[idx] = self.need
				idx += 1

			idx = self.tags_info.p_tag_start
			while idx < self.tags_info.x_tag_start and idx < self.tags_info.p_tag_start + self.p_max:
				re[idx] = self.need
				idx += 1


			if self.stack[-1] == 14:
				pass
			else:
				re[self.prev_variable] = self.mask
			return re
		else:
			re = self._get_zeros(self.tags_info.tag_size)
			re[1] = self.need
			return re
		
	def update(self, ix):
		if ix < self.tags_info.tag_size:
			if ix == 1:
				pass
			if ix >= 3 and ix < self.tags_info.p_rel_start:
				self.stack.append(ix)
				if ix==4:
					self.stack_drs.append(ix)
				self.prev_prev_variable = self.prev_variable
				self.prev_variable = -1
			elif ix >= self.tags_info.p_rel_start and ix < self.tags_info.p_tag_start:
				self.stack.append(ix)
				self.prev_prev_variable = self.prev_variable
				self.prev_variable = -1
			elif ix == 2:
				if self.stack[-1] == 4:
					self.stack_drs.pop()
				self.stack.pop()
				if len(self.stack)>0:
					while self.stack[-1] == 17:
						self.stack.pop()
						self.stack.pop()
				self.prev_prev_variable = self.prev_variable
				self.prev_variable = -1
			else:
				if ix >= self.tags_info.p_tag_start and ix < self.tags_info.x_tag_start:
					pass
				elif ix >= self.tags_info.x_tag_start and ix < self.tags_info.e_tag_start:
					assert self.x >= ix - self.tags_info.x_tag_start + 1
					if self.x == ix - self.tags_info.x_tag_start + 1:
						self.x += 1
				elif ix >= self.tags_info.e_tag_start and ix < self.tags_info.s_tag_start:
					assert self.e >= ix - self.tags_info.e_tag_start + 1
					if self.e == ix - self.tags_info.e_tag_start + 1:
						self.e += 1
				elif ix >= self.tags_info.s_tag_start and ix < self.tags_info.t_tag_start:
					assert self.s >= ix - self.tags_info.s_tag_start + 1
					if self.s == ix - self.tags_info.s_tag_start + 1:
						self.s += 1
				elif ix >= self.tags_info.t_tag_start and ix < self.tags_info.t10_tag_start:
					assert self.t >= ix - self.tags_info.t_tag_start + 1
					if self.t == ix - self.tags_info.t_tag_start + 1:
						self.t += 1
				elif ix >= self.tags_info.t10_tag_start and ix < self.tags_info.x10_tag_start:
					assert self.t10 >= ix - self.tags_info.t10_tag_start + 1
					if self.t10 == ix - self.tags_info.t10_tag_start + 1:
						self.t10 += 1
				elif ix >= self.tags_info.x10_tag_start and ix < self.tags_info.tag_size:
					assert self.x10 >= ix - self.tags_info.x10_tag_start + 1
					if self.x10 == ix - self.tags_info.x10_tag_start + 1:
						self.x10 += 1
				else:
					assert False
				self.prev_prev_variable = self.prev_variable
				self.prev_variable = ix

		else:
			self.stack.append(ix)
			self.prev_prev_variable = self.prev_variable
			self.prev_variable = -1

	def _print_state(self):
		print("stack", self.stack)
		print("xes", self.x, self.e, self.s,self.t,self.t10,self.x10)

	def _get_zeros(self, size):
		return [self.mask for i in range(size)]

	def _get_ones(self, size):
		return [self.need for i in range(size)]
