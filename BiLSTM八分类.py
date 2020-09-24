# -*- coding: utf-8 -*-
"""
@author: Zhichen Ren
"""
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np

#read files class
class ReadFilesName():
    def __init__(self, path):
        self.path = path
        
     # read files
    def ReadFileName(self):
        dataset = pd.read_csv(self.path,index_col = 0)       
        return dataset
    
    #print dataset
    def printData(self,dataset):
        print("print dataset:")
        print(dataset.head)

#preprocess class      
class Preprocess():
    def __init__(self, dataset):
       
        self.dataset = pd.DataFrame(dataset)
    
    #delete column
    def deleteColumn(self,columnName):
        
        self.dataset = self.dataset.drop(columnName, axis=1, inplace = False)
        return self.dataset
    
    #delete nan value
    def deleteNaN(self):
        self.dataset = self.fillna(0, inplace = False)
        
        return self.dataset
    
    #get data
    def getdata(self):
        
        return self.dataset
    
    #print data
    def printData(self):
        self.dataset.set_option('display.max_rows', None)
        print (self.dataset)
        
#BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, v_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional=True, drop_prob= 0.5):
        super(BiLSTM, self).__init__()
        self.output_size = output_size

        self.n_layers = n_layers

        self.hidden_dim = hidden_dim

        self.bidirectional = bidirectional
        
        # embedding and LSTM layers        
        self.embedding = nn.Embedding(v_size, embedding_dim)        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=bidirectional)
        
        # dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # linear and sigmoid layers        
        if bidirectional:          
            self.fc = nn.Linear(hidden_dim*2, output_size)        
        else:          
            self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()
        
    def forward(self, x, hidden):
        batch_size =x.size(0)
        
        # embeddings and lstm_out
        x = x.long()

        embeds = self.embedding(x)

        lstm_out, hidden =self.lstm(embeds, hidden)
        
        # dropout and fully-connected layer
        out = self.dropout(lstm_out)

        out = self.fc(out)

        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first        
        sig_out = sig_out.view(batch_size, -1)  
        
        # get last batch of labels
        sig_out = sig_out[:, -1]     
        
        # return last sigmoid output and hidden state        
        return sig_out, hidden
    
    def init_hidden(self, batch_size):        
               
        # Create two new tensors         
        # initialized to zero, for hidden state and cell state of LSTM        
        weight = next(self.parameters()).data
        number = 1

        if self.bidirectional:

           number = 2
        if (train_on_gpu):            
             hidden = (weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().cuda(), weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_().cuda())        
        else:            
             hidden = (weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_(),weight.new(self.n_layers*number, batch_size, self.hidden_dim).zero_() )                
             
             
        return hidden
  
        
        
#main method
        
if __name__=="__main__":
    
    path = 'C:\\Users\\Jason Chen\\Desktop\\BLSTMtest\\PET_AV45_NoNANWithReduceLabel.csv'
    readfile = ReadFilesName(path)
    dataset = readfile.ReadFileName()
    readfile.printData(dataset)
    
    pre = Preprocess(dataset) 
    pre = pre.getdata()
    print(pre.head)
    
    #preprocess delete useless values
    pre = pre.drop('VISCODE', axis = 1, inplace =False)
    pre = pre.drop('DX_bl', axis = 1, inplace =False)
    pre = pre.drop('DXCHANGE', axis = 1, inplace =False)
    pre = pre.drop('AGE', axis = 1, inplace =False)
    
    #delete nan values
    pre = pre.fillna(0, inplace = False)

    data= pre    
    values = data.values
    values = values.astype('float32')
    
    #create feature and label
    feature = values[:, 1:]
    label = values[:,0]
    print("")
    print("features:")
    print (feature)
    print("")
    print("labels:")
    print(label)
    print("")
      
    split= 0.8
    split_idx = int(len(feature)*split)
    train_x, remaining_x = feature[:split_idx], feature[split_idx:]
    train_y, remaining_y = label[:split_idx], label[split_idx:]
    test_idx = int(len(remaining_x)*0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]
    
    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),"\nValidation set: \t{}".format(val_x.shape),"\nTest set: \t\t{}".format(test_x.shape))
    print("")
    print("train_x:")
    print(train_x)
    print("train_y:")
    print(train_y)
    print("val_x:")
    print(val_x)
    print("val_y:")
    print(val_y)
    print("test_x:")
    print(test_x)
    print("test_y:")
    print(test_y)
    

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    #dataloaders
    batch_size=50
    #make sure the shuffle your training data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last= True)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size,drop_last= True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size,drop_last= True)
    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next() 
    print('Sample input size: ', sample_x.size()) # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size()) # batch_size
    print('Sample label: \n', sample_y)
    
    #checking if GPU is available
    train_on_gpu= torch.cuda.is_available()
    if (train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, traing on CPU.')
        
    vocab_size =len(train_x)*len(train_x[0])    
    output_size = 1
    embedding_dim =64
    hidden_dim=256    
    n_layers = 2
    bidirectional = True  
    net = BiLSTM(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional)
    print(net)

    
    # loss and optimization functions

    lr=0.001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

   # training params
   
    epochs = 4 

    print_every = 100

    # gradient clipping
    clip=5 

    # move model to GPU, if available
    if(train_on_gpu):

        net.cuda()
    net.train()

    # train for some number of epochs
    for e in range(epochs):

        # initialize hidden state

        h = net.init_hidden(batch_size)

        counter = 0

        # batch loop
        for inputs, labels in train_loader:

            counter += 1
            if(train_on_gpu):

                inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state
            h = tuple([each.data for each in h])

        # zero accumulated gradients

            net.zero_grad()
        # get the output from the model
            output, h = net(inputs, h)
            
        # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)

            optimizer.step()
        # loss stats
            if counter % print_every == 0:
            # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:

                # Creating new variables for the hidden state

                    val_h = tuple([each.data for each in val_h])
                    if(train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()
                    output, val_h = net(inputs, val_h)

                    val_loss = criterion(output.squeeze(), labels.float())
 
                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e+1, epochs),"Step: {}...".format(counter),"Loss: {:.6f}...".format(loss.item()),"Val Loss: {:.6f}".format(np.mean(val_losses)))

    # Get test data loss and accuracy
    # track loss
    test_losses = []     
    num_correct = 0

    # init hidden state    
    h = net.init_hidden(batch_size)
    net.eval()
    
    # iterate over test data
    
    for inputs, labels in test_loader:
    
        # Creating new variables for the hidden state
        h = tuple([each.data for each in h])
    
        if(train_on_gpu):
    
            inputs, labels = inputs.cuda(), labels.cuda()
    

        # get predicted outputs
    
        output, h = net(inputs, h)
   
        # calculate loss
    
        test_loss = criterion(output.squeeze(), labels.float())   
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        # rounds to the nearest integer
        pred = torch.round(output.squeeze())     
        # compare predictions to true label
    
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())    
        num_correct += np.sum(correct)
    
    # -- stats! -- ##
    
    # avg test loss
    
    print("Test loss: {:.3f}".format(np.mean(test_losses)*(-1)))

    # accuracy over all test data
    test_acc = num_correct/len(test_loader.dataset)
    
    print("Test accuracy: {:.3f}".format(test_acc))

            
            
            
            
            
            
            