'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.

In our example we will be using data that can be downloaded at:
https://www.kaggle.com/tongpython/cat-and-dog

In our setup, it expects:
- a data/ folder
- train/ and validation/ subfolders inside data/
- cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-X in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 0-X in data/train/dogs
- put the dog pictures index 1000-1400 in data/validation/dogs

We have X training examples for each class, and 400 validation examples
for each class. In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''


import torchvision 
import torch.nn as nn 
import torch 
import torch.nn.functional as F 
from torchvision import transforms,models,datasets 

import matplotlib.pyplot as plt 
from PIL import Image 
import numpy as np 
from torch import optim 
import csv
from collections import OrderedDict


def imshow(inp, title=None): 
    """Imshow for Tensor.""" 
    inp = inp.numpy().transpose((1, 2, 0)) 
    plt.figure(figsize=(20,150)) 
    plt.imshow(inp) 
	

def main():
	train_data_dir = '/content/example-versioning/data/train' 
	transform = transforms.Compose([transforms.Resize(255), 
								  transforms.CenterCrop(224), 
								  transforms.ToTensor()]) 
	dataset = torchvision.datasets.ImageFolder(train_data_dir, transform= transform) 
	
	test_data_dir = '/content/example-versioning/data/validation' 
	transform = transforms.Compose([transforms.Resize(255), 
								  transforms.CenterCrop(224), 
								  transforms.ToTensor()]) 
	test_dataset = torchvision.datasets.ImageFolder(test_data_dir, transform= transform) 	

	classes = ('cat', 'dog')	
	SEED = 1

	# CUDA?
	cuda = torch.cuda.is_available()
	print("CUDA Available?", cuda)

	# For reproducibility
	torch.manual_seed(SEED)

	if cuda:
		torch.cuda.manual_seed(SEED)
		
	# dataloader arguments - something you'll fetch these from cmdprmt
	dataloader_args = dict(shuffle=True, batch_size=16, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=8)

	train_loader = torch.utils.data.DataLoader(dataset, **dataloader_args) 

	test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args) 

	# inputs, classe = next(iter(train_loader)) 
	# # Make a grid from batch 
	# out = torchvision.utils.make_grid(inputs, scale_each= True) 
	# imshow(out) 

	model = models.vgg16(pretrained=True)

	for params in model.parameters(): 
		params.requires_grad = False 


	classifier = nn.Sequential(OrderedDict([ 
		('fc1',nn.Linear(25088,4096)), 
		('relu1',nn.ReLU()), 
		('dropout1', nn.Dropout(p=0.5)),
		('fc2',nn.Linear(4096,4096)), 
		('relu2',nn.ReLU()), 
		('dropout2', nn.Dropout(p=0.5)),
		('fc3',nn.Linear(4096,2)), 
		('Output',nn.LogSoftmax(dim=1)) 
	])) 

	model.classifier = classifier 	

	use_cuda = torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	print(device)
	model = model.to(device)

	optimizer= optim.Adam(model.classifier.parameters()) 
	criterian= nn.NLLLoss() 

	with open('metrics.csv', 'w') as fd:
	  csv_out = csv.writer(fd)
	  csv_out.writerow(['Epoch', 'Train loss','Test loss',
						'Train accuracy', 'Test accuracy',
						'Accurarcy of Cat', 'Accurarcy of dog'
						])
	  fd.close()
	  
	  
	list_train_loss = [] 
	list_test_loss = [] 
	list_train_accuracy = [] 
	list_test_accuracy = [] 


	for epoch in range(10): 
		train_loss = 0 
		test_loss = 0 
		train_accuracy = 0 
		test_accuracy = 0

		misclassified_images = []  
		list_class_accuracy = []
		correct_pred = {classname: 0 for classname in classes}
		total_pred = {classname: 0 for classname in classes}

		for bat,(img,label) in enumerate(train_loader): 
			# moving batch and labels to gpu 
			img = img.to(device) 
			label = label.to(device) 

			model.train() 
			optimizer.zero_grad() 
			output = model(img) 

			loss = criterian(output,label) 
			train_loss += loss.item()  
			loss.backward() 
			optimizer.step() 

			ps = torch.exp(output) 
			top_ps,top_class = ps.topk(1,dim=1) 
			equality = top_class == label.view(*top_class.shape) 
			train_accuracy += torch.mean(equality.type(torch.FloatTensor)).item()  
		
		with torch.no_grad():
		  for bat,(img,label) in enumerate(test_loader): 
			  img = img.to(device) 
			  label = label.to(device) 

			  model.eval() 
			  logps= model(img) 
			  loss = criterian(logps,label) 
			  test_loss+= loss.item() 

			  ps=torch.exp(logps) 
			  top_ps,top_class=ps.topk(1,dim=1) 
			  equality=top_class == label.view(*top_class.shape) 
			  test_accuracy += torch.mean(equality.type(torch.FloatTensor)).item() 

			  # collect the correct predictions for each class
			  for label, prediction in zip(label, top_class):
				  if label == prediction:
					  correct_pred[classes[label]] += 1
				  total_pred[classes[label]] += 1          
			  
	 

		list_train_loss.append(train_loss/len(train_loader.dataset)) 
		list_test_loss.append(test_loss/len(test_loader)) 
		list_train_accuracy.append(train_accuracy/len(train_loader.dataset)) 
		list_test_accuracy.append(test_accuracy/len(test_loader))

		print('epoch: ',epoch,' train_loss: ',train_loss/len(train_loader.dataset),
			  ' test_loss: ',test_loss/len(test_loader.dataset),' train accuracy: ', train_accuracy/len(train_loader),
			  'test_aacurarcy: ', test_accuracy/len(test_loader) ) 
		
		# print accuracy for each class
		for classname, correct_count in correct_pred.items():
			accuracy = 100 * float(correct_count) / total_pred[classname]
			list_class_accuracy.append(accuracy)
			print("Accuracy for class {:5s} is: {:.1f} %".format(classname,accuracy))

		with open('metrics.csv', 'a') as fd:
		  csv_out = csv.writer(fd)
		  csv_out.writerow([epoch, train_loss/len(train_loader.dataset),test_loss/len(test_loader.dataset),
							train_accuracy/len(train_loader),test_accuracy/len(test_loader),
							list_class_accuracy[0], list_class_accuracy[1]
							])
		  fd.close()  

	torch.save(model,'model.h5')


if __name__ == "__main__":
    main()	
