import numpy as np
import torch, torchvision


class TrainingTask(torch.nn.Module):
    def __init__(self, basemodule, epochs=25, lr=0.1, callback=None):
        super().__init__()
        self.basemodule        = basemodule
        self.epochs            = epochs
        self.lr                = lr
        self.progress_callback = callback
    
    def training_step(self, batch):
        raise NotImplementedError()
    
    def validation_step(self, batch):
        raise NotImplementedError()
    
    def configure_optimizers(self):
        #optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        optim = torch.optim.SGD(self.parameters(), self.lr, momentum=0.9, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, self.epochs, eta_min=self.lr/100)
        return optim, sched
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def train_one_epoch(self, loader, optimizer, scheduler=None):
        for i,batch in enumerate(loader):
            loss,logs  = self.training_step(batch)
            logs['lr'] = optimizer.param_groups[0]['lr']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.callback.on_batch_end(logs, i, len(loader))
        if scheduler:
            scheduler.step()
    
    def eval_one_epoch(self, loader):
        for i,batch in enumerate(loader):
            logs  = self.validation_step(batch)
            self.callback.on_batch_end(logs, i, len(loader))
    
    def fit(self, loader_train, loader_valid=None, epochs='auto'):
        self.epochs = epochs
        if epochs == 'auto':
            self.epochs = max(15, 50 // len(loader_train))
            
        if self.progress_callback is not None:
            self.callback = TrainingProgressCallback(self.progress_callback, self.epochs)
        else:
            self.callback = PrintMetricsCallback()
        
        optim, sched  = self.configure_optimizers()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.cuda.empty_cache()
        try:
            self.to(device)
            self.__class__.stop_requested = False
            for e in range(self.epochs):
                if self.__class__.stop_requested:
                    break
                self.train().requires_grad_(True)
                self.train_one_epoch(loader_train, optim, sched)
                
                self.eval().requires_grad_(False)
                if loader_valid:
                    self.eval_one_epoch(loader_valid)
                
                self.callback.on_epoch_end(e)
        except KeyboardInterrupt:
            print('\nInterrupted')
        except Exception as e:
            #prevent the exception getting to ipython (memory leak)
            import traceback
            traceback.print_exc()
        finally:
            self.zero_grad(set_to_none=True)
            self.eval().cpu().requires_grad_(False)
            torch.cuda.empty_cache()
     
    #XXX: class method to avoid boiler code
    @classmethod
    def request_stop(cls):
        cls.stop_requested = True



class PrintMetricsCallback:
    '''Prints metrics after each training epoch in a compact table'''
    def __init__(self):
        self.epoch = 0
        self.logs  = {}
        
    def on_epoch_end(self, epoch):
        self.epoch = epoch + 1
        self.logs  = {}
        print() #newline
    
    def on_batch_end(self, logs, batch_i, n_batches):
        self.accumulate_logs(logs)
        percent     = ((batch_i+1) / n_batches)
        metrics_str = ' | '.join([f'{k}:{float(np.mean(v)):>9.5f}' for k,v in self.logs.items()])
        print(f'[{self.epoch:04d}|{percent:.2f}] {metrics_str}', end='\r')
    
    def accumulate_logs(self, newlogs):
        for k,v in newlogs.items():
            self.logs[k] = self.logs.get(k, []) + [v]

class TrainingProgressCallback:
    '''Passes training progress as percentage to a custom callback function'''
    def __init__(self, callback_fn, epochs):
        self.n_epochs    = epochs
        self.epoch       = 0
        self.callback_fn = callback_fn
    
    def on_batch_end(self, logs, batch_i, n_batches):
        percent     = ((batch_i+1) / (n_batches*self.n_epochs))
        percent    += self.epoch / self.n_epochs
        self.callback_fn(percent)
    
    def on_epoch_end(self, epoch):
        self.epoch = epoch + 1

