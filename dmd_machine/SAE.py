#The python code of the stacked autoencoder model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Lambda,
    Dropout,
    Flatten,
    Reshape,
    Conv2DTranspose,
)
from tensorflow.keras.models import load_model, Sequential, Model
class StackAutoencoder(object):
	"""docstring for Stack_Autoencoder"""
	def __init__(self):
		super(StackAutoencoder, self).__init__()


	def save_model(self, model, model_name, save_dir):
	# function for saving model

		if not os.path.isdir(save_dir):
			os.makedirs(save_dir)
		model_path = os.path.join(save_dir, model_name)
		model.save(model_path)


	def encoder_model(self, input_dim, encoding_dim):

		#encoded = Dense(encoding_dim * 32, activation='relu')(input_dim)
		#encoded = Dense(encoding_dim * 16, activation='relu')(encoded)
		#encoded = Dense(encoding_dim * 8, activation='relu')(encoded)
		encoded = Dense(encoding_dim * 8, activation='relu')(input_dim)
		encoded = Dense(encoding_dim * 4, activation='relu')(encoded)
		encoded = Dense(encoding_dim * 2, activation='relu')(encoded)
		code = Dense(encoding_dim)(encoded)

		return code

	def decoder_model(self, vtu_data, code, encoding_dim):

		decoded = Dense(encoding_dim * 2, activation='relu')(code)
		decoded = Dense(encoding_dim * 4, activation='relu')(decoded)
		decoded = Dense(encoding_dim * 8, activation='relu')(decoded)
		#decoded = Dense(encoding_dim * 16, activation='relu')(decoded)
		#decoded = Dense(encoding_dim * 32, activation='relu')(decoded)
		decoded = Dense(vtu_data.shape[1], activation='sigmoid')(decoded)

		return decoded


	def load_model(self, vtu_data, encoding_dim):

		input_dim = Input(shape=(vtu_data.shape[1]))
		encoder_output = self.encoder_model(input_dim, encoding_dim)
		self.encoder = Model(inputs=input_dim, outputs=encoder_output, name='encoder') # from input to code
		# self.encoder.summary()


		# Create a encoder model 
		decoder_input = Input(shape=(encoding_dim)) #the input of the encoder 
		decoder_output = self.decoder_model(vtu_data, decoder_input, encoding_dim)
		self.decoder = Model(inputs=decoder_input, outputs=decoder_output, name="decoder")
		# self.decoder.summary()


		# connect the encoder and decoder model
		encoder_img = self.encoder(input_dim) # call the encoder model
		decoder_img = self.decoder(encoder_img) #call the decoder model
		self.autoencoder = Model(inputs=input_dim, outputs=decoder_img, name='autoencoder') #construct the autoencoder model

	def train_model(self, train, validation, test, epochs, batch_size, model_save_folder, encoder_file_name, decoder_file_name, AE_file_name):

		check_model = ModelCheckpoint(model_save_folder + '/' + AE_file_name,
									monitor='val_loss',
									save_best_only=True,
									verbose=1)
		reduce_LR = ReduceLROnPlateau(monitor='val_loss',
									factor=0.5,
									patience=5,
									verbose=1,
									mode='min',
									min_delta=1e-10,
									cooldown=0,
									min_lr=0)

		self.history_record = self.autoencoder.fit(train, train,
                        epochs = epochs,
                        batch_size = batch_size,
                        callbacks=[check_model, reduce_LR],
                        validation_data=(validation, validation))

		# draw_Acc_Loss(self.history_record)
		self.save_model(self.encoder, encoder_file_name, model_save_folder)
		self.save_model(self.decoder, decoder_file_name, model_save_folder)
		self.save_model(self.autoencoder, AE_file_name, model_save_folder)
		self.save_model(self.autoencoder, AE_file_name, model_save_folder)

		print(" DeepAE model trained successfully")

		scores = self.autoencoder.evaluate(test, test, batch_size, verbose=1)

		print('Test loss:', scores[0], '\nTest accuracy:', scores[1])
