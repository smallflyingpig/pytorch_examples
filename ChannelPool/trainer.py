import torch
import os
import numpy as np
from utils import progress_bar


class Trainer(object):
    def __init__(self, model, trainloader, valloader, optimizer, device, criterion, logger, writer, save_model_dir):
        self.model, self.trainloader, self.valloader, self.optimizer, self.device, self.criterion, self.logger, self.writer = \
            model, trainloader, valloader, optimizer, device, criterion, logger, writer
        self.save_model_dir = save_model_dir
        self.best_acc = 0

    def train(self, total_epoch, val_interval, lr_scheduler=None, start_epoch=0, best_acc=0):
        self.best_acc = best_acc
        for epoch_idx in range(start_epoch, total_epoch):
            if lr_scheduler is not None:
                lr_scheduler.step()
            self.train_once(epoch_idx)
            if epoch_idx % val_interval == 0 or epoch_idx == total_epoch-1:
                acc = self.eval(epoch_idx)
                self.update_best_acc(acc, epoch=epoch_idx)
        
            
            
    def train_once(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
    
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            self.logger.debug("epoch:{} | Batch:{} | Type:{} | Loss:{} | Acc:{}% ({:d}/{:d})".format(
                epoch, batch_idx, 'train', train_loss/(batch_idx+1), 100.*correct/total, correct, total
            ))
        train_loss = train_loss/total
        correct = 100. * correct/total
        self.writer.add_scalars('loss', {'train_loss':train_loss}, global_step=epoch)
        self.writer.add_scalars('accu', {'train_accu':correct}, global_step=epoch)
        
        
    def eval(self, epoch, log=True):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
    
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                if log:
                    progress_bar(batch_idx, len(self.valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                    self.logger.debug("epoch:{} |Batch:{} | Type:{} | Loss:{} | Acc:{}% ({:d}/{:d})".format(
                    epoch, batch_idx, 'test', test_loss/(batch_idx+1), 100.*correct/total, correct, total
                    ))
        test_loss = test_loss/total
        correct = 100. * correct/total
        if log:
            self.writer.add_scalars('loss', {'test_loss':test_loss}, global_step=epoch)
            self.writer.add_scalars('accu', {'test_accu':correct}, global_step=epoch)
    
        # Save checkpoint.
        acc = 100.*correct/total
        return acc
        
    
    def test(self, loop_num=1, dataloader=None):
        if dataloader is None:
            dataloader = self.valloader
        accu_list = []
        for idx in range(loop_num):
            accu_list.append(self.eval(idx, log=False))
        accu_np = np.array(accu_list)
        try:
            best, mean, std = np.max(accu_np), np.mean(accu_np), np.std(accu_np)
        except ValueError as e:
            print(e)
            print(accu_np)
        return {'best':best, 'mean':mean, 'std':std}

    
    def update_best_acc(self, acc, epoch):
        if acc > self.best_acc:
            print('Saving..')
            state = {
                'net': self.model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(self.save_model_dir, 'ckpt.t7'))
            self.best_acc = acc
    