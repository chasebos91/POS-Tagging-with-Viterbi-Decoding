import numpy as np
from collections import defaultdict

import warnings

warnings.simplefilter("ignore")


class HMM:
	def __init__(self, validation):
		self.dictionary = {}
		self.transitions = defaultdict(lambda: defaultdict(int))
		self.emissions = defaultdict(lambda: defaultdict(int))
		self.context = {}
		self.validation = validation

	def init_model(self):
		#build vocab
		with open("POS_train.pos", "r") as file:
			for l in file:
				if l.split():
					w, t = l.split("\t")
					if w not in self.dictionary:
						self.dictionary[w] = 1
					else:
						self.dictionary[w] += 1

		self.dictionary["UNK"] = 0
		self.dictionary["*ENDSENTENCE*"] = 0
		self.testdata = self.read_files("POS_test.words")

		# get emission/transition counts

		with open("POS_train.pos", "r") as file:
			curr = "*STOP*"
			for l in file:
				if l.split():
					w, t = l.split()
					if w not in self.dictionary:
						w = "UNK"
				else:
					w = "*ENDSENTENCE*"
					t = "*STOP*"

				self.transitions[curr][t] += 1
				self.emissions[t][w] += 1
				if t not in self.context:
					self.context[t] = 1
				else:
					self.context[t] += 1
				curr = t

			self.tags = list(self.context.keys())
			self.build_matrices()

	def build_matrices(self):
		# transition matrix
		K = len(self.tags)
		N = len(self.dictionary)
		self.A = np.zeros((K, K))

		for r in range(K):
			for c in range(K):
				i = 0
				curr = self.tags[r]
				t = self.tags[c]
				if curr in self.transitions and t in self.transitions[curr]:
					i = self.transitions[curr][t]
				self.A[r][c] = i / (self.context[curr] * K)

		# emission matrix
		self.B = np.zeros((K, N))

		for i, tag in enumerate(self.tags):
			for j, word in enumerate(self.dictionary):
				temp = 0

				if word in self.emissions[tag]:
					temp = self.emissions[tag][word]

				self.B[i][j] = (temp + 1e-5) / (self.context[tag] * N)

	def read_files(self, file):
		data = []

		with open(file, "r") as f:

			for i, w in enumerate(f):
				if w.split():
					if w.strip() in self.dictionary:
						data.append(w.strip())
					else: data.append("UNK")
				else: data.append("*ENDSENTENCE*")

		return data

	def read_tags(self, file):
		data = []

		with open(file, "r") as f:
			for i, t in enumerate(f):
				if t.split():
						w, tag = t.split()
						data.append(tag.strip())
				else: data.append("*STOP*")

		return data

	def decode(self):

		if self.validation:
			data = self.read_files("POS_dev.words")
			data = data
			labels = self.read_tags("POS_dev.pos")
			labels = labels
			x = self.viterbi(data)
			count = 0
			true = 0
			for p, t in zip(x, labels):
				if p == t: true += 1
				count += 1
			print("Validation accuracy: ", true / count)
		else:

			test_preds = self.viterbi(self.testdata)
			self.output_testfile(test_preds, self.testdata, "POS_test.pos")

	def output_testfile(self, preds, data, file):
		with open(file, "w") as f:
			for i in range(len(preds)):
				f.write((data[i] +"\t" + preds[i] + "\n"))
			f.close()

	def viterbi(self, data):

		K = len(self.tags)
		T = len(data)

		VM = np.zeros((K, T))
		BpM = np.zeros((K, T))

		# init prior
		idx = self.tags.index("*STOP*")
		for i in range(K):
			if self.A[idx][i] != 0:
				VM[i][0] = self.A[idx][i] * self.B[i][list(self.dictionary.keys()).index(data[0])]

		# compute MAP probabilities for each word in data and keep track of most probable path
		for t in range(T):
			if t % 50 == 0: print(str(t) + " steps completed")
			for s in range(K):
				bestpathprob = - np.inf
				bestpath = None
				for k in range(K):
					temp = VM[k][t-1] + np.log(self.A[k][s]) + np.log(self.B[s][list(self.dictionary.keys()).index(data[t])])
					if temp > bestpathprob:
						bestpath = k
						bestpathprob = temp
				VM[s][t] = bestpathprob
				BpM[s][t] = bestpath

		#Backtrack BpM to construct most probable path and associated tags
		path = np.zeros(T)
		path[-1] = np.argmax(VM[:, T - 1])
		preds = [""] * T
		preds[-1] = self.tags[int(path[T - 1])]

		for i in range(T - 1, 0, -1):
			path[i - 1] = BpM[int(path[i])][i]
			preds[i - 1] = self.tags[int(path[i - 1])]

		return preds


#################  MAIN  ##############################


val = input("Enter V for validation or T for Test: ") == "V"
hmm = HMM(val)
hmm.init_model()
#pickle.dump(hmm, open("HMM.sav", "wb"))
#hmm = pickle.load(open("HMM.sav", "rb"))
hmm.decode()
