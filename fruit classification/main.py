# Fruit Classification with a CNN

from model import FNet
# import required modules
from utils import dataset
import torch 
import torch.tensor
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt


def train_model(dataset_path, debug=False, destination_path='', save=False):
    
    """Trains model with set hyper-parameters and provide an option to save the model.

	This function should contain necessary logic to load fruits dataset and train a CNN model on it. It should accept dataset_path which will be path to the dataset directory. You should also specify an option to save the trained model with all parameters. If debug option is specified, it'll print loss and accuracy for all iterations. Returns loss and accuracy for both train and validation sets.

	Args:
		dataset_path (str): Path to the dataset folder. For example, '../Data/fruits/'.
		debug (bool, optional): Prints train, validation loss and accuracy for every iteration. Defaults to False.
		destination_path (str, optional): Destination to save the model file. Defaults to ''.
		save (bool, optional): Saves model if True. Defaults to False.

	Returns:
		loss (torch.tensor): Train loss and validation loss.
		accuracy (torch.tensor): Train accuracy and validation accuracy.
	"""
	# Write your code here
	# The code must follow a similar structure
	# NOTE: Make sure you use torch.device() to use GPU if available

    # device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #dataset_path='E:\E-Yantra\2018-19\Task 1\Task 1B\Data\fruits'
    #destination_path = dataset_path

    df, df_train, df_test = dataset.create_and_load_meta_csv_df(dataset_path=dataset_path, destination_path=destination_path, randomize=True, split=0.8)

    data_transforms = {
        'train': transforms.Compose([transforms.ToTensor()]),
        'test': transforms.Compose([transforms.ToTensor()])
    }

    image_datasets = {'train': dataset.ImageDataset(df_train, transform=data_transforms['train']), 
                    'test': dataset.ImageDataset(df_test, transform=data_transforms['test'])}

    # make data loaders
    dataloaders = { 'train_loader': torch.utils.data.DataLoader(image_datasets['train'],batch_size=7,shuffle=True),
                    'test_loader': torch.utils.data.DataLoader(image_datasets['test'],batch_size=7,shuffle=True) }


    cnn=FNet().to(device) #create model called cnn
    loss_func = nn.CrossEntropyLoss() #create Cross Entropy loss function
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001) #Use Adam optimizer with learning rate=0.001

    epoch = 2 #no of epochs

    for i in range(epoch): 
        print("epoch: ", i+1)
        num_times=0
        losses=[] #array to store loss after every 100 batches
        for j,[image1,label1] in enumerate(dataloaders['train_loader']): #get image and label from train_loader
            image = Variable(image1) 
            label = Variable(label1)

            optimizer.zero_grad()
            result = cnn.forward(image1) #use cnn model to get result
            loss = loss_func(result,label1) # use cross entrpy loss function to calculate loss
            loss.backward() #use back propagation
            optimizer.step()

            if j % 100 == 0: 
                print(j,'\t::: loss ---',loss.item()) #after 100 batches print loss
                losses.append(loss.item()) #append loss to array
                num_times += 1
            
            if j % 100 == 0:
                cnn.eval() #use cnn.eval() function to calculate accuracy 
                correct_train = correct_valid = 0 
                total_train = total_valid = 0

                # loop to calculate accuracy after each epoch of train_loader i.e test dataset
                for image2,label2 in dataloaders['train_loader']: 
                    image2 = Variable(image2)
                    result2 = cnn(image2) #use cnn to get result

                    _,pred2 = torch.max(result2.data,1) # use to find max score of a neuron w.r.t. 1

                    total_train += label2.size(0) 
                    correct_train += (pred2 == label2).sum().item() #no of correct indentified labels 
                    acc_train = (correct_train/total_train)*100 #calculate accuracy train dataset 

                print("Accuracy of Train Data : ",acc_train) #print accuracy of train dataset

                for image3,label3 in dataloaders['test_loader']: # loop to calculate accuracy after each epoch of test_loader i.e test dataset
                        image3 = Variable(image3)
                        result3 = cnn(image3) #use cnn to get result

                        _,pred3 = torch.max(result3.data,1) # use to find max score of a neuron w.r.t. 1

                        total_valid += label3.size(0) 
                        correct_valid += (pred3 == label3).sum().item() #no of correct indentified labels 
                        acc_valid = (correct_valid/total_valid)*100 #calculate accuracy of test dataset

                print("Accuracy of Validation Data : ",acc_valid) #print accuracy of test dataset
        plt.plot([i for i in range(num_times)],losses,label='epoch'+str(i)) #plot for loss and update after each epoch 
        plt.legend(loc=1,mode='expanded',shadow=True,ncol=2)


    plt.show() #plt graph for loss 


    #plt.show()
    if (save==True):
        torch.save(cnn.state_dict(),'cnn.ckpt')
        print("Model Saved") 




if __name__ == "__main__":
	train_model('../Data/fruits/', debug=True, save=True, destination_path='./')
