from . import training_base, models
import numpy as np
import torch, torchvision



class Stage1_Task(training_base.TrainingTask):
    def __init__(self, *args, seg_model:torch.nn.Module, box_size=64, dsc_size=16, **kwargs):
        super().__init__(*args,**kwargs)
        #segmentation model in list to hide parameters
        self.seg_model  = [seg_model.eval().requires_grad_(False)]
        self.box_size   = box_size
        self.dsc_size   = dsc_size
    
    def training_step(self, batch):
        x     = batch
        #general augmentation
        x     = color_jitter(random_rotate_flip(x))
        #positive augmentation
        x_pos = color_jitter(x)
        #negative augmentation
        x_neg = random_rotate_flip(x)
        
        #torch.cat because of batchnorm
        y     = self.basemodule(torch.cat([x,x_pos,x_neg]), return_features=True)
        y0, y_pos, y_neg = torch.split(y, [len(_x) for _x in [x,x_pos,x_neg]])
        
        with torch.no_grad():
            self.seg_model[0].to(x.device)
            seg_out = self.seg_model[0].eval()(x)     > 0.5
            seg_neg = self.seg_model[0].eval()(x_neg) > 0.5
        
        pts     = sample_points(seg_out, n=4)
        pts_neg = sample_points(seg_neg, n=8)
        
        dsc0    = models.extract_descriptors(y0,    pts,     self.box_size, self.dsc_size)
        dsc_pos = models.extract_descriptors(y_pos, pts,     self.box_size, self.dsc_size)
        dsc_neg = models.extract_descriptors(y_neg, pts_neg, self.box_size, self.dsc_size)
        
        sims_pos = models.compute_descriptor_similarities(dsc0, dsc_pos)  # NxN similarity matrix
        sims_pos = sims_pos[torch.eye(len(dsc0)).bool()]                  # N i-to-i similarities
        loss_pos = -torch.log( sims_pos.clamp(min=1e-6)).mean()
        
        sims_neg = models.compute_descriptor_similarities(dsc0, dsc_neg)
        loss_neg = -torch.log((1-sims_neg).clamp(min=1e-6)).mean()
        
        loss     = loss_pos + loss_neg
        logs     = {
            'loss'      : loss.item(),
            'loss_pos'  : loss_pos.item(),
            'loss_neg'  : loss_neg.item(),
        }
        return loss, logs


class Stage2_Task(training_base.TrainingTask):
    def __init__(self, *args, box_size=64, dsc_size=16, **kwargs):
        super().__init__(*args,**kwargs)
        self.box_size   = box_size
        self.dsc_size   = dsc_size
    
    def training_step(self, batch):
        x0,x1, p0,p1    = batch
        
        #augmentations
        #make sure both images and points are rotated the same number of times
        tk       = choose_random_rotate_flip()
        x0,p0    = random_rotate_flip(x0,p0,tk)
        x1,p1    = random_rotate_flip(x1,p1,tk)
        
        x0       = color_jitter(x0)
        x1       = color_jitter(x1)
        
        #torch.cat because of batchnorm
        y        = self.basemodule(torch.cat([x0,x1]), return_features=True)
        y0, y1   = torch.split(y, [len(_x) for _x in [x0,x1]])
        
        dsc0     = models.extract_descriptors(y0,    p0,     self.box_size, self.dsc_size)
        dsc1     = models.extract_descriptors(y1,    p1,     self.box_size, self.dsc_size)
        
        sims_pos = models.compute_descriptor_similarities(dsc0, dsc1)     # NxN similarity matrix
        sims_pos = sims_pos[torch.eye(len(dsc0)).bool()]                  # N i-to-i similarities
        loss_pos = -torch.log( sims_pos.clamp(min=1e-6)).mean()
        
        #reshape (b*n)chw -> bnchw
        dsc0     = dsc0.reshape(p0.shape[0], p0.shape[1], *dsc0.shape[1:])
        dsc1     = dsc1.reshape(p1.shape[0], p1.shape[1], *dsc1.shape[1:])
        sims_neg = torch.einsum('bnchw,bmchw->bnm', dsc0, dsc1)  / self.dsc_size**2   #-1...+1
        sims_neg = sims_neg/2+0.5                                                     # 0...+1
        sims_neg = zero_out_if_close(sims_neg, p0)
        loss_neg = -torch.log(torch.clamp_min(1-sims_neg, 1e-6)).mean()
        
        loss     = loss_pos + loss_neg
        logs     = {
            'loss'      : loss.item(),
            'loss_pos'  : loss_pos.item(),
            'loss_neg'  : loss_neg.item(),
        }
        return loss, logs



def sample_points(segmap:torch.Tensor, n=64) -> [torch.Tensor]:
    '''Sample n random points, preferably where segmap is positive'''
    assert len(segmap.shape) == 4  #BxCxHxW
    pts = []
    for sm in segmap:
        sm   = sm[0] > 0.5
        #get points where sm is positive
        yx0  = torch.stack(torch.where(sm), -1)
        yx0  = yx0[torch.randperm(len(yx0))[:n]]
        #fill with random points if not enough
        yx1  = np.random.randint(sm.shape, size=(n-len(yx0),2) )
        yx   = torch.cat([yx0, torch.as_tensor(yx1).to(yx0.device)])
        pts.append(yx)
    return pts

def zero_out_if_close(x_mat, pts_list:list, threshold=8):
    x_mat     = x_mat.clone()
    for i,pts in enumerate(pts_list):
        dist_mat = torch.abs(pts[:,None] - pts[None]).sum(-1)   #L1 distance
        ok_mat   = (dist_mat > threshold).to(x_mat.device)
        x_mat[i] *= ok_mat
    return x_mat


###AUGMENTATIONS###

def choose_random_rotate_flip():
    choices = [('rot', 1), ('rot', 2), ('rot', 3), ('flip', -1), ('flip', -2)]
    return choices[np.random.randint(len(choices))]

def random_rotate_flip(x, p=None, tk=None):
    if tk is None:
        tk = choose_random_rotate_flip()
    t,k = tk
    if t=='flip':
        x = torch.flip(x, dims=[k])
        p = flip_points(p, x.shape[-2:], axis=k)
    elif t == 'rot':
        x = torch.rot90( x, k, dims=[-2,-1] )
        p = rot90_points(p, x.shape[-2:], k=k)
    return (x) if p is None else (x,p)

color_jitter = torchvision.transforms.ColorJitter(brightness=(0.5,1.7), contrast=(0.7,1.7), saturation=0.5, hue=0.3)

def rot90_points(p, shape, k=1):
    if   p is None: return None
    if   k==0:      return p
    elif k==1:      return torch.stack([ shape[1]-1-p[...,1],            p[...,0] ], -1)
    elif k==2:      return torch.stack([ shape[0]-1-p[...,0], shape[1]-1-p[...,1] ], -1)
    elif k==3:      return torch.stack([            p[...,1], shape[0]-1-p[...,0] ], -1)

def flip_points(p, shape, axis):
    if   p is None:         return None
    if   axis in [0, -2]:   return torch.stack([ shape[0]-1-p[...,0],            p[...,1] ], -1)
    elif axis in [1, -1]:   return torch.stack([            p[...,0], shape[1]-1-p[...,1] ], -1)


