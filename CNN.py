# import the necessary packages
import time
import torch
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch import flatten, nn
from torch.optim import Adam,SGD,RMSprop,Adagrad,Adadelta
from torchshape import tensorshape
import DataPreprocessing
from EarlyStopper import EarlyStopper
import torch.cuda.amp.autocast_mode
import torch.cuda.amp.grad_scaler


class CNN_Model(Module):
	def init_loss_function(self,function_name):
		if function_name == "cross-entropy":
			self.loss_fn = nn.CrossEntropyLoss()
		elif function_name == "mse":
			self.loss_fn = nn.MSELoss()
		elif function_name == "L1":
			self.loss_fn = nn.L1Loss()
		elif function_name == "binary-cross-entropy":
			self.loss_fn = nn.BCELoss()
		elif function_name == "neg-log-likelihood":
			self.loss_fn = nn.NLLLoss()

	def init_optimizer(self,optimizer_name=None,learning_rate=None,momentum=None,weight_decay=None):
		if optimizer_name=="adam":
			self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
		elif optimizer_name=="SGD":
			self.opt = SGD(self.parameters(), lr=learning_rate,momentum=momentum,weight_decay=weight_decay)
		elif optimizer_name=="RMSprop":
			self.opt = RMSprop(self.parameters(), lr=learning_rate,weight_decay=weight_decay)
		elif optimizer_name=="Adagrad":
			self.opt = Adagrad(self.parameters(), lr=learning_rate,weight_decay=weight_decay)
		elif optimizer_name=="Adadelta":
			self.opt = Adadelta(self.parameters(), lr=learning_rate,weight_decay=weight_decay)


	def __init__(self, numChannels=3, classes=1,loss_function="cross-entropy",optimizer="adam",learning_rate=0.001,device=None,input_shape=None,architecture=None,early_stop=True,waiting=100,min_delta=0,momentum=1,weight_decay=1e-5):
		# call the parent constructor
		super(CNN_Model, self).__init__()

		layers = []
		first_layer = True
		last_output_shape = 0
		for layer_name, params in architecture.items():
			#print(last_output_shape)
			if 'Conv2d' in layer_name:
				if first_layer:
					first_layer = False
					layers.append(Conv2d(in_channels=numChannels, out_channels=params['out_channels'],kernel_size=params['kernel_size'],padding='same'))
					last_output_shape = tensorshape(layers[-1], input_shape)
				else:
					layers.append(Conv2d(in_channels=last_output_shape[1], out_channels=params['out_channels'],kernel_size=params['kernel_size'],padding='same'))
					last_output_shape = tensorshape(layers[-1], last_output_shape)
			elif 'activation' in layer_name:
				layers.append(params['function'])
			elif 'MaxPool' in layer_name:
				layers.append(nn.MaxPool2d(kernel_size=params['kernel_size'], stride=params['stride']))
				last_output_shape = tensorshape(layers[-1], last_output_shape)
			elif 'Linear' in layer_name:
				if layer_name == list(architecture.keys())[-2]:
					layers.append(Linear(in_features=last_output_shape[1], out_features=classes))
				else:
					layers.append(Linear(in_features=last_output_shape[1], out_features=params['out_features']))
					last_output_shape = tensorshape(layers[-1], last_output_shape)
			elif 'Flatten' in layer_name:
				layers.append(nn.Flatten())
				last_output_shape = tensorshape(layers[-1], last_output_shape)
			elif 'Dropout' in layer_name:
				layers.append(params['function'])
				last_output_shape = tensorshape(layers[-1], last_output_shape)
			elif 'BatchNorm' in layer_name:
				if 'BatchNorm1d' in layer_name:
					layers.append(nn.BatchNorm1d(last_output_shape[1]))
					last_output_shape = tensorshape(layers[-1], last_output_shape)
				elif 'BatchNorm2d' in layer_name:
					layers.append(nn.BatchNorm2d(last_output_shape[1]))
					last_output_shape = tensorshape(layers[-1], last_output_shape)
				else:
					print('Wrong batch normalization definition...')
					exit(1)


		self.arch_layers = nn.ModuleList(layers)

		for m in self.arch_layers:
			if isinstance(m, nn.Conv2d):
				torch.nn.init.xavier_uniform_(m.weight)


		self.init_loss_function(loss_function)
		self.init_optimizer(optimizer_name=optimizer,learning_rate=learning_rate,momentum=momentum,weight_decay=weight_decay)
		self.verbose_levels = [0, 1, 10, 100, 1000]
		self.History = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "epoch_time": []}
		self.device = device
		self.early_stop = early_stop
		if self.early_stop:
			self.early_stopper = EarlyStopper(waiting=waiting, mind=min_delta)

	def save_model(self):
		torch.save(self.state_dict(), "saved_cnn_model.pth")

	def forward(self, x):
		for _, layer in enumerate(self.arch_layers, start=0):
			x = layer(x)
		return x

	def evaluate(self,xDataLoader):
		accuracy = 0
		plot_index = 0
		with torch.no_grad():
			self.eval()
			for (x, y) in xDataLoader:
				(x, y) = (x.to(self.device), y.to(self.device))
				predicted = self(x)
				accuracy += self.calculate_accuracy(predicted, y)
				if plot_index == 0:
					plot_index += 1
					DataPreprocessing.plot_images(x.cpu().detach().numpy(),y.cpu().detach().numpy(),pred=predicted.cpu().detach().numpy())
		return accuracy/len(xDataLoader)

	def calculate_accuracy(self,pred,true):
		return torch.sum(pred.argmax(1) == true.argmax(1))/len(true)

	def print_progress(self, phase, accuracy, loss, current_epoch,verbose):
		if current_epoch % self.verbose_levels[verbose] == 0:
			print("Phase " + str(phase) + " - Accuracy: " + str(accuracy * 100) + "%" + " - " + "loss: " + str(loss))

	def fit(self,num_epochs=10,trainDataLoader=None,valDataLoader=None,verbose=1):
		lossFn = self.loss_fn
		scaler = torch.cuda.amp.grad_scaler.GradScaler()
		# loop over our epochs
		for epoch in range(0, num_epochs):
			start = time.time()
			if (verbose != 0):
				if (epoch+1) % self.verbose_levels[verbose] == 0:
					print("\n")
					print("Epoch: " + str(epoch+1) + "/" + str(num_epochs) + " - ║{0:20s}║ {1:.1f}%".format(
						'🟩' * int((epoch+1) / num_epochs * 20), (epoch+1) / num_epochs * 100))
			# set the model in training mode
			self.train()
			total_train_loss = 0
			total_val_loss = 0
			total_train_accuracy = 0
			total_val_accuracy = 0
			# loop over the training set
			for (x, y) in trainDataLoader:
				# send the input to the device
				(x, y) = (x.to(self.device), y.to(self.device))

				with torch.cuda.amp.autocast_mode.autocast(dtype=torch.float16):
					# perform a forward pass and calculate the training loss
					predicted = self(x)
					loss = lossFn(predicted, y)
				# zero out the gradients, perform the backpropagation step,
				# and update the weights
				self.opt.zero_grad(set_to_none=True)
				scaler.scale(loss).backward()
				#self.opt.step()
				scaler.step(optimizer=self.opt)
				# add the loss to the total training loss so far and
				# calculate the number of correct predictions

				scaler.update()

				total_train_loss += loss
				accuracy = self.calculate_accuracy(predicted,y)
				total_train_accuracy += accuracy
			self.print_progress("Train",(total_train_accuracy/len(trainDataLoader)).item(),(total_train_loss/len(trainDataLoader)).item(),epoch+1,verbose)

			# switch off autograd for evaluation
			with torch.no_grad():
				# set the model in evaluation mode
				self.eval()
				# loop over the validation set
				for (x, y) in valDataLoader:
					# send the input to the device
					(x, y) = (x.to(self.device), y.to(self.device))

					# make the predictions and calculate the validation loss
					with torch.cuda.amp.autocast_mode.autocast(dtype=torch.float16):
						predicted = self(x)
						loss = lossFn(predicted, y)
					total_val_loss += loss
					accuracy = self.calculate_accuracy(predicted,y)
					total_val_accuracy += accuracy
				self.print_progress("Val", (total_val_accuracy/len(valDataLoader)).item(),(total_val_loss/len(valDataLoader)).item(), epoch + 1,verbose)
			end = time.time()
			print("Epoch time elapsed: "+str((end-start)))
			self.History["loss"].append((total_train_loss.cpu().detach().numpy()/len(trainDataLoader)))
			self.History["accuracy"].append((total_train_accuracy.cpu().detach().numpy()/len(trainDataLoader)))
			self.History["val_loss"].append((total_val_loss.cpu().detach().numpy()/len(valDataLoader)))
			self.History["val_accuracy"].append((total_val_accuracy.cpu().detach().numpy()/len(valDataLoader)))
			self.History["epoch_time"].append(end-start)

			if self.early_stop:
				if self.early_stopper.early_stop((total_val_loss/len(valDataLoader)).item()):
					break
		self.save_model()








