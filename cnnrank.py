from scorer import Scorer
import numpy as np
import logging
import pdb, os, inflect, argparse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Flatten
from keras.layers import Convolution1D, MaxPooling1D, Merge, GlobalMaxPooling1D
from keras.models import Model
from keras.utils import np_utils
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
EMB_DIM = 300
MAX_NB_WORDS=20000
MAX_LEN = 5000
nb_filter = 5
batch_size = 30
epochs = 3
TRAIN = 0.7
logging.basicConfig(level=logging.INFO)
class NNScorer(Scorer):
	def __init__(self, topics, pairs, pairstest, mode):

		super().__init__(topics, pairs, pairstest, mode)
		#self.loadWord2Vec('/home/Btech13/sumit.cs13/GoogleNews-vectors-negative300.bin', True)

	def get_single_instances(self, filename, fmap):
		X = []
		with open(filename) as fin:
			for line in fin:
				l = line.strip().split('\t')
				try:
					idx = 3
					text = ''
					fm = fmap
					while fm > 0:
						take = fm % 2
						fm = int(fm / 2)
						if take == 1:
							text += l[idx]
						idx += 1
					X.append([l[0], l[1], text])
				except:
					pdb.set_trace()
		return X

	def get_train_instances(self, X):
		sz = len(X)
		split = int(sz/2)
		# take half as they are
		X_train = X[0:split]
		y = [1] * len(X_train)
		i = split
		# assign random profession to other half
		# TODO incorporate some kind of profession dependency
		while i < sz:
			j = np.random.randint(0, len(self.topics))
			if self.topics[j] == X[i][1]:
				continue
			samp = X[i]
			X_train.append([samp[0], self.topics[j], samp[2]])
			y.append(0)
			i = i + 1
		return np.asarray(X_train), y

	def fit_data(self, X, y):
		texts = X[:,2].ravel().tolist()
		entities = X[:,1].ravel().tolist()
		self.MAX_LEN = len(max(texts, key = lambda x:len(x)))
		self.MAX_LENE = len(max(entities, key = lambda x:len(x)))
		self.tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
		self.tokenizer.fit_on_texts(texts+entities)
		self.word_index = self.tokenizer.word_index
		self.X_train = []
		print(self.MAX_LEN)
		self.X_train = []
		f1 = pad_sequences(self.tokenizer.texts_to_sequences(X[:,2]), MAX_LEN)
		#f2 = pad_sequences(self.tokenizer.texts_to_sequences(X[:,1]), self.MAX_LENE)
		self.X_train = np.asarray(f1)

	def get_train_test(self):
		t = int(len(self.X_train) * TRAIN)
		return self.X_train[0:t], self.y[0:t], self.X_train[t:], self.y[t:]

	def get_embedding_matrix(self):
		embedding_matrix = np.zeros((len(self.word_index) + 1, 300))
		for word, i in self.word_index.items():
			try:
				embedding_vector = self.w2v[word]
				embedding_matrix[i] = embedding_vector
			except Exception as e:
				pass
		return embedding_matrix
	
	def get_data(self, train = True):
		pairs = self.pairs
		if not train:
			pairs = self.pairstest
		X = np.empty((len(pairs), EMB_DIM * 2))
		y = []
		for idx, pair in enumerate(pairs):
			vec1 = self.get_vector(pair[0])
			vec2 = self.get_vector(pair[1])
			v = np.append(vec1, vec2)
			X[idx] = v
			y.append(pair[2])
		return X, y

	def build_model(self):
		X = self.get_single_instances('data/profession.one.sorted', 0b11)
		X, y = self.get_train_instances(X)
		self.fit_data(X,y)
		train_X, train_y, test_X, test_y = self.get_train_test()
		#self.loadWord2Vec('/home/Btech13/sumit.cs13/GoogleNews-vectors-negative300.bin', True)
		embedding_matrix = self.get_embedding_matrix()
		# answer model
		branches = []
		train_xs = []
		dev_xs = []
		input_shape = (MAX_LEN, EMB_DIM)
		context_input = Input(shape=input_shape)
		emb = Embedding(len(self.word_index), EMB_DIM, input_length=MAX_LEN, name="embedding")(context_input)
		conv_blocks = []
		for sz in [2,3]:
			conv = Convolution1D(
						filters=nb_filter,
						padding='valid',
						activation='relu',
						subsample_length=1)
			conv = GlobalMaxPooling1D()(conv)
			conv_blocks.append(conv)
		z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
		
		# define input for entity vector
		ent_input = Input(shape=(300,), dtype='float32', name='ent_input')
		# merge entity vector input with CNN output on context
		x = merge([z, ent_input], mode='concat', name='merged_layer')
		# Stack a fully connected deep network
		x = Dense(64, activation='relu', name='dense_one')(x)
		loss = Dense(1, activation='sigmoid', name='output')(x)
		model = Model(input=[context_input, ent_input], outputs=loss)
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		print(model.get_layer('merged_layer').output_shape)

	def get_vector(self, word):
		vec = np.zeros((1,EMB_DIM))
		if not self.w2v:
			return vec
		words = word.split()
		for w in words:
			try:
				vec += self.w2v[w]
			except:
				pass
		return vec / len(word)
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", help="Input directory", action="store_true")
	parser.add_argument("input", help="Input directory value")

	parser.add_argument("-o", help="Output directory", action="store_true")
	parser.add_argument("output", help="Output directory value")
	parser.add_argument("-pro", help="profession flag", action="store_true")
	parser.add_argument("-nat", help="nationality flag", action="store_true")
	args = parser.parse_args()

	input = ""
	output = ""
	if args.i:
		input = args.input
	if args.o:
		output = args.output
	if args.pro:
		logging.info("Getting topics")
		topics = Scorer.getTopics(input+'/professions')
		logging.info("Getting pairs")
		pairs = NNScorer.getPersonPairs(input+'/profession.train')
		pairstest = NNScorer.getPersonPairs(input+'/profession.test')
		nns = NNScorer(topics, pairs, pairstest, 'profession')
		nns.build_model()
if __name__ == "__main__":
	main()
