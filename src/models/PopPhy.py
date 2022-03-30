import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score
from utils.popphy_io import get_stat, get_stat_dict
from utils.feature_map_analysis import get_feature_map_rankings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class PopPhyCNN():

	def __init__(self, input_size, num_class, model_name,config):
		# model_name:topo-cnn\topo-gcn
		num_row, num_col = input_size
		self.model_name=model_name
		self.num_kernel = int(config.get('PopPhy', 'NumberKernel'))
		self.kernel_height = int(config.get('PopPhy', 'KernelHeight'))
		self.kernel_width = int(config.get('PopPhy', 'KernelWidth'))
		self.num_fc_nodes = int(config.get('PopPhy', 'NumFCNodes'))
		self.num_cnn_layers = int(config.get('PopPhy', 'NumConvLayers'))
		self.num_fc_layers = int(config.get('PopPhy', 'NumFCLayers'))
		self.lamb = float(config.get('PopPhy', 'L2Lambda'))
		self.drop = float(config.get('PopPhy', 'Dropout'))

		reg = tf.keras.regularizers.l2(self.lamb)
		self.model = tf.keras.Sequential()

		#加入权重W层

		#加入非线性relu层
		# self.model.add(tf.keras.layers.Dense(num_fea_gcnw, input_shape=(num_row, num_col), activation='relu', kernel_regularizer=reg, bias_regularizer=reg,
		# 									 name="gcnW"))
		if self.model_name=='topo-cnn':
			self.model.add(tf.keras.layers.Reshape((num_row, num_col, 1),input_shape=(num_row, num_col)))
			# self.model.add(tf.keras.layers.GaussianNoise(0.01, input_shape=(num_row, num_col, 1)))
			self.model.add(tf.keras.layers.GaussianNoise(0.01))
			for i in range(0, self.num_cnn_layers):
				self.model.add(tf.keras.layers.Conv2D(filters=self.num_kernel, kernel_size=(self.kernel_height, self.kernel_width),
					activation='relu', bias_regularizer=reg, kernel_regularizer=reg, name="conv_"+str(i)))
				# 循环一次，一个Conv2D层，filters=32，kernel_size=3*10,acti=relu,
				self.model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
		else:
			self.model.add(tf.keras.layers.GaussianNoise(0.01, input_shape=(num_row, num_col)))
			for i in range(0, self.num_cnn_layers):
				self.model.add(
					tf.keras.layers.Conv1D(filters=self.num_kernel, kernel_size=self.kernel_height, activation='relu',
										   kernel_initializer='glorot_uniform',
										   kernel_regularizer=reg, bias_regularizer=reg, name="conv_" + str(i)))
				self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

		# for i in range(0, num_cnn_layers):
		# 	self.model.add(tf.keras.layers.Conv2D(filters=num_kernel, kernel_size=(kernel_height, kernel_width),
		# 		activation='relu', bias_regularizer=reg, kernel_regularizer=reg, name="conv_"+str(i)))
		# 	# 循环一次，一个Conv2D层，filters=32，kernel_size=3*10,acti=relu,
		# 	self.model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
			# 一个最大池化层

        #网上一维卷积模板
		# for i in range(0, num_cnn_layers):
		# 	self.model.add(tf.keras.layers.Conv1D(filters=num_kernel, kernel_size=kernel_height, strides=1, padding='valid', data_format='channels_last',
		# 					dilation_rate=1, activation='relu', use_bias=True, kernel_initializer='glorot_uniform',
		# 					bias_initializer='zeros', kernel_regularizer=reg, bias_regularizer=reg,
		# 					name="conv_"+str(i)))
		# 	self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

		self.model.add(tf.keras.layers.Flatten())
		self.model.add(tf.keras.layers.Dropout(self.drop))

		for i in range(0, self.num_fc_layers):
			self.model.add(tf.keras.layers.Dense(self.num_fc_nodes, activation='relu', kernel_regularizer=reg, bias_regularizer=reg, name="fc_"+str(i)))
			self.model.add(tf.keras.layers.Dropout(self.drop))

		self.model.add(tf.keras.layers.Dense(num_class, activation='softmax', kernel_regularizer=reg, bias_regularizer=reg, name="output"))

		self.patience = int(config.get('PopPhy', 'Patience'))
		self.learning_rate = float(config.get('PopPhy', 'LearningRate'))
		self.batch_size = int(config.get('PopPhy', 'BatchSize'))


	def train(self, train, train_weights=[]):
		train_x, train_y = train
		# train_x = np.expand_dims(train_x, -1)

		def mcc_metric(y_true, y_pred):
			predicted = tf.cast(tf.greater(y_pred, 0.5), tf.float32)
			true_pos = tf.math.count_nonzero(predicted * y_true)
			true_neg = tf.math.count_nonzero((predicted - 1) * (y_true - 1))
			false_pos = tf.math.count_nonzero(predicted * (y_true - 1))
			false_neg = tf.math.count_nonzero((predicted - 1) * y_true)
			x = tf.cast((true_pos + false_pos) * (true_pos + false_neg) * (true_neg + false_pos) * (true_neg + false_neg), tf.float32)
			return tf.cast((true_pos * true_neg) - (false_pos * false_neg), tf.float32) / tf.sqrt(x)
		
		def auc_metric(y_true, y_pred):
			return tf.numpy_function(roc_auc_score, (y_true, y_pred), tf.double)

		self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss='categorical_crossentropy', 
			metrics=[auc_metric, mcc_metric])

		if len(np.unique(train_y) == 2):
			es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=self.patience, restore_best_weights=True)
		else:
			es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=self.patience, restore_best_weights=True)

		if len(train_weights) == 0:
			self.model.fit(train_x, train_y, batch_size=self.batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1)
			self.model.fit(train_x, train_y, batch_size=self.batch_size, verbose=0, epochs=10)
		else:
			self.model.fit(train_x, train_y, batch_size=self.batch_size, verbose=0, epochs=1000, callbacks=[es_cb], validation_split=0.1, sample_weight = train_weights)
			self.model.fit(train_x, train_y, batch_size=self.batch_size, verbose=0, epochs=10, sample_weight = train_weights)
		return

	def test(self, test):
		test_x, test_y = test
		# if self.model_name == 'topo-cnn':
		# 	test_x = np.expand_dims(test_x, -1)
		preds = self.model.predict(test_x)
		stats = get_stat_dict(test_y, preds)
		parame = {}
		parame['num_cnn_layers'] = self.num_cnn_layers
		parame['num_kernel'] = self.num_kernel
		parame['kernel_height'] = self.kernel_height
		if self.model_name == 'topo-cnn':
			parame['kernel_width'] = self.kernel_width
		parame['lamb'] = self.lamb
		parame['drop'] = self.drop
		parame['num_fc_layers'] = self.num_fc_layers
		parame['num_fc_nodes'] = self.num_fc_nodes
		parame['learning_rate'] = self.learning_rate
		parame['batch_size'] = self.batch_size
		parame['patience'] = self.patience

		return preds, stats,parame
	
	def get_feature_map_weights(self):
		w = self.model.get_layer(name="conv_0").get_weights()
		return w

	def get_feature_maps(self, x):
		fm = []
		try:
			fm_model = tf.keras.Model(inputs=self.model.input, outputs=self.model.get_layer("conv_0").output)
			fm = fm_model.predict(x)
		except:
			Print("Conv Layer Not Found!!!!")
		return fm

	def get_feature_scores(self, train, g, label_set, features, config):
		scores = []
		x, y = train
		# x = np.expand_dims(x,-1)
		pred = self.model.predict(x)
		fm = self.get_feature_maps(x)
		pred_label = np.argmax(pred, axis=1)
		true_label = np.argmax(y, axis=1)
		w, b = self.get_feature_map_weights()
		out_scores = get_feature_map_rankings(x, true_label, pred_label, fm, w, b, g, label_set, features, config)

		for l in label_set:
			scores.append(out_scores[l].reindex(features).values)

		scores = np.array(scores)
		scores = np.transpose(scores,(2,1,0))
		scores = np.squeeze(scores)
		return scores

	def destroy(self):
		tf.keras.backend.clear_session()
		return
