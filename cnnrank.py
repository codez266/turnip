from scorer import Scorer
import numpy as np
import logging
import theano
import ipdb
import pdb, os, inflect, argparse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input
from keras.layers import Embedding, Flatten, concatenate, Reshape
from keras.layers import Convolution1D, MaxPooling1D, Merge, GlobalMaxPooling1D, Convolution2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers.merge import Concatenate, Dot
from keras.layers import merge
from keras.models import Model
from keras.utils import np_utils
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from keras.models import model_from_json
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
EMB_DIM = 300
MAX_NB_WORDS=20000
MAX_LEN = 1000
nb_filter = 10
batch_size = 200
epochs = 4
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
					X.append([l[0], l[1], text[100:]])
				except:
					ipdb.set_trace()
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
		k = list(zip(X_train, y))
		np.random.shuffle(k)
		k = list(zip(*k))
		X_train = k[0]
		y = k[1]
		return np.asarray(X_train), y

	def vectorize_persons(self, pairs):
		data = []
		y = []
		for p in pairs:
			if 'opening_text' in self.persons[p[0]]:
				data.append([p[0], p[1], self.persons[p[0]]['opening_text']])
				y.append(p[2])
		X = self.text_to_vectors(np.asarray(data))
		return X, np.asarray(y, dtype=np.int32)

	def text_to_vectors(self, X):
		f1 = pad_sequences(self.tokenizer.texts_to_sequences(X[:,2]), self.MAX_LEN)
		f2 = pad_sequences(self.tokenizer.texts_to_sequences(X[:,1]), self.MAX_LENE)
		return [f1,f2]

	def fit_data(self, X, y):
		texts = X[:,2].ravel().tolist()
		entities = X[:,1].ravel().tolist()
		#self.MAX_LEN = len(max(texts, key = lambda x:len(x)))
		self.MAX_LEN = MAX_LEN
		self.MAX_LENE = len(max(entities, key = lambda x:len(x)))
		self.tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
		self.tokenizer.fit_on_texts(texts+entities)
		self.word_index = self.tokenizer.word_index
		self.X_train = []
		self.logger.info("Found %s unique tokens", self.MAX_LEN)
		self.X_train = []
		f1 = pad_sequences(self.tokenizer.texts_to_sequences(X[:,2]), self.MAX_LEN)
		f2 = pad_sequences(self.tokenizer.texts_to_sequences(X[:,1]), self.MAX_LENE)
		#f = lambda x: self.get_vector(x)
		#f2 = np.array(list(map(f, X[:,1])))
		#for idx, x in enumerate(X):
		#	self.X_train.append(np.array([f1[idx], f2[idx]]))
		#ipdb.set_trace()
		self.X_train = [f1,f2]
		self.y = np.array(y)

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
		self.loadWord2Vec('/home/Btech13/sumit.cs13/GoogleNews-vectors-negative300.bin', True)
		embedding_matrix = self.get_embedding_matrix()
		lim = int(len(self.X_train[0])*TRAIN)
		input_shape = (self.MAX_LEN, EMB_DIM)
		context_input = Input(shape=(self.MAX_LEN,))
		emb = Embedding(len(self.word_index) + 1, EMB_DIM, input_length=self.MAX_LEN, weights=[embedding_matrix],name="embedding")(context_input)
		emb = Dropout(0.2)(emb)
		conv_blocks = []
		for sz in [2,3]:
			conv = Convolution1D(
						filters=nb_filter,
						kernel_size=sz,
						padding='valid',
						activation='relu',
						strides=1,
						input_shape=input_shape)(emb)
			conv = GlobalMaxPooling1D()(conv)
			#conv = Flatten()(conv)
			conv_blocks.append(conv)
		z = Concatenate(name='conv_layer')(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
		#z = Dense(64, activation='relu', name='conv_dense')(z)

		# define input for entity vector
		#ent_inp = Input(shape=(300,), dtype='float32', name='ent_input')
		ent_inp = Input(shape=(self.MAX_LENE,), dtype='float32', name='ent_input')
		ent_emb = Embedding(len(self.word_index) + 1, EMB_DIM, input_length=self.MAX_LENE, weights=[embedding_matrix],name="ent_embedding")(ent_inp)
		conv_ent = Convolution1D(
					filters=3,
					kernel_size=2,
					padding='valid',
					activation='relu',
					strides=1,
					input_shape=input_shape)(ent_emb)
		conv_ent = GlobalMaxPooling1D()(conv_ent)
		#ent = Dense(64, activation='relu', name='ent_dense')(ent_inp)
		# merge entity vector input with CNN output on context
		x = concatenate([z, conv_ent], name='merged_layer')
		#x = Dot(1 , name='merged_layer')([z, ent])
		# Stack a fully connected deep network
		#x = Dense(512, activation='relu', name='dense_one')(x)
		#x = Dropout(0.4)(x)
		x = Dense(128, activation='relu', name='dense_two')(x)
		x = Dropout(0.2)(x)
		loss = Dense(1, activation='sigmoid', name='output')(x)
		model = Model(inputs=[context_input, ent_inp], outputs=loss)
		model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
		print(model.get_layer('conv_layer').output_shape)
		print(model.get_layer('merged_layer').output_shape)
		model.fit([self.X_train[0][0:lim], self.X_train[1][0:lim]], self.y[:lim],
			batch_size=batch_size,
			epochs=epochs,
			validation_data=([self.X_train[0][lim:], self.X_train[1][lim:]], self.y[lim:])
			)
		ipdb.set_trace()
		model_json = model.to_json()
		with open("cnn_model.json", "w") as json_file:
			json_file.write(model_json)
		model.save_weights("cnn_model.h5")
		print("Saved model to disk")

	def test_cnn_model(self, model_name):
		X = self.get_single_instances('data/profession.one.sorted', 0b11)
		X, y = self.get_train_instances(X)
		self.fit_data(X,y)
		#self.loadWord2Vec('/home/Btech13/sumit.cs13/GoogleNews-vectors-negative300.bin', True)
		#embedding_matrix = self.get_embedding_matrix()
		lim = int(len(self.X_train[0])*TRAIN)
		
		json_file = open(model_name+'.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		model.load_weights(model_name+'.h5')
		print("Loaded model from disk")
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		ipdb.set_trace()
		labels = model.predict([self.X_train[0][lim:], self.X_train[1][lim:]])
		print('predictions done')
		truth = self.y[lim:]
		t_l = int(len(truth)*TRAIN)
		truth_train = truth[:t_l]
		truth_test = truth[t_l:]
		clf=None
		score=0
		print('fitting Logistic regression')
		for i, C in enumerate((100, 1, 0.01)):
		# turn down tolerance for short training time
			clf_l1_LR = LogisticRegression(C=C, penalty='l1', tol=0.01)
			clf_l2_LR = LogisticRegression(C=C, penalty='l2', tol=0.01)
			clf_l1_LR.fit(labels[:t_l], truth_train)
			clf_l2_LR.fit(labels[:t_l], truth_train)
			l1=clf_l1_LR.score(labels[t_l:],truth_test)
			l2=clf_l2_LR.score(labels[t_l:],truth_test)
			if l1 > score:
				clf = clf_l1_LR
				score = l1
			if l2 > score:
				clf = clf_l2_LR
				score = l2
			print(l1, " ", l2)
		preds = clf.predict(labels[t_l:]).astype(int)
		y_test = truth_test.astype(int)
		print(classification_report(y_test, preds))
		ipdb.set_trace()

	def regression(self, model_name):
		X = self.get_single_instances('data/profession.one.sorted', 0b11)
		X, y = self.get_train_instances(X)
		self.fit_data(X,y)
		lim = int(len(self.X_train[0])*TRAIN)
		
		json_file = open(model_name+'.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		model.load_weights(model_name+'.h5')
		print("Loaded model from disk")
		model.layers.pop()
		model_new = Model(model.input, model.layers[-1].output)
		model_new.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		
		self.persons = Scorer.getWikipediaTexts2(self.persons)
		X_train, y_train = self.vectorize_persons(self.pairs)

		X_test, y_test = self.vectorize_persons(self.pairstest)
		X2_train = model_new.predict([X_train[0], X_train[1]])
		X2_test = model_new.predict([X_test[0], X_test[1]])
		est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
					random_state=0, loss='lad').fit(X2_train, y_train)
		pairstest = np.asarray(self.pairstest)
		classes = est.predict(X2_test)
		ipdb.set_trace()
		pairstest = np.hstack((pairstest, np.array([classes])))
		print(mean_squared_error(y_test, classes))



	def get_vector(self, word):
		vec = np.zeros((EMB_DIM,))
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
		#nns.test_cnn_model('cnn_model')
		#nns.regression('cnn_model')
if __name__ == "__main__":
	main()
