from FocalLoss import focal_loss
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable

class DML(object):
    def __init__(self, models, optimizer, parallel = False):
        self.models = models
        self.optimizer = optimizer
        self.loss_kl = nn.KLDivLoss(reduction='batchmean').cuda()  
        self.loss_fc = focal_loss(gamma = 3, alpha = [800/243, 800/293, 800/264]).cuda() 
        self.model_num = len(models)
        if parallel:
            for i in range(self.model_num):
                self.models[i] = nn.DataParallel(self.models[i])
    
    def train(self, n_epochs, train_loader, valid_loader, val_loss_min = np.Inf, resume = False):       
        if resume:
            for i in range(self.model_num):
                checkpoint = torch.load('DML' + str(i) + '.pth', map_location='cpu')
                self.models[i].load_state_dict(checkpoint, False)
        for i in range(self.model_num):
            self.models[i] = self.models[i].cuda()
            
        for epoch in range(1, n_epochs+1):
            print('running epoch: {}'.format(epoch))
            train_loss = self.train_one_epoch(epoch, train_loader)
            val_loss = self.validate(epoch, valid_loader)
            print('\tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                train_loss, val_loss))
            if val_loss <= val_loss_min:
                print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                val_loss_min,
                val_loss))
                for i in range(self.model_num):
                    torch.save(self.models[i].state_dict(), 'DML' + str(i) + '.pth')
                val_loss_min = val_loss
            
        writer.add_scalar("mango/train_loss", train_loss, epoch)
        writer.add_scalar("mango/val_loss", val_loss, epoch)
    
    def train_one_epoch(self, epoch, train_loader):
        train_loss = 0.0
        for i in range(self.model_num):
            self.models[i].train()
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            outputs=[]
            for model in self.models:
                outputs.append(model(data))
            self.optimizer.zero_grad()
            for i in range(self.model_num):
                fc_loss = self.loss_fc(outputs[i], target)
                kl_loss = 0
                for j in range(self.model_num):
                    if i!=j:
                        kl_loss = kl_loss + self.loss_kl(F.log_softmax(outputs[i], dim = 1), 
                                                F.softmax(Variable(outputs[j]), dim=1))
                loss = fc_loss + kl_loss / (self.model_num - 1)
                
                loss.backward()
                train_loss += loss.item()*data.size(0)
                
            self.optimizer.step()
        train_loss = train_loss/len(train_loader.dataset)    
                
        return train_loss

    def validate(self, epoch, valid_loader):
        valid_loss = 0.0
        for data, target in valid_loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)            
            outputs=[]
            for model in self.models:
                outputs.append(model(data))
            for i in range(self.model_num):
                fc_loss = self.loss_fc(outputs[i], target)
                kl_loss = 0
                for j in range(self.model_num):
                    if i!=j:
                        kl_loss = kl_loss + self.loss_kl(F.log_softmax(outputs[i], dim = 1), 
                                                F.softmax(Variable(outputs[j]), dim=1))
                loss = fc_loss + kl_loss / (self.model_num - 1)
                
                valid_loss += loss.item()*data.size(0)
        valid_loss = valid_loss/len(valid_loader.dataset)
        return valid_loss

    def test(self, loader):
        valid_loss = 0.0
        for i in range(self.model_num):
            checkpoint = torch.load('DML' + str(i) + '.pth', map_location='cpu')
            self.models[i].load_state_dict(checkpoint, False)
            self.models[i].cuda()
            self.models[i].eval()
        correct = 0  
        total = 0
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)              
            outputs=[]
            for model in self.models:
                outputs.append(model(data))
            for i in range(self.model_num):
                fc_loss = self.loss_fc(outputs[i], target)
                kl_loss = 0
                for j in range(self.model_num):
                    if i!=j:
                        kl_loss = kl_loss + self.loss_kl(F.log_softmax(outputs[i], dim = 1), 
                                                F.softmax(Variable(outputs[j]), dim=1))
                loss = fc_loss + kl_loss / (self.model_num - 1)
                valid_loss += loss.item()*data.size(0)
            pred =  outputs[0] + outputs[1]
            pred = pred.data.max(1, keepdim=True)[1]
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)
        valid_loss = valid_loss/len(loader.dataset)
        print('Test Loss: {:.6f}'.format(valid_loss))

        print('Test Accuracy: %2d%% (%2d/%2d)' % (
            100. * correct / total, correct, total))