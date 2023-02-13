import os, typing as tp, datetime
import numpy as np
import PIL.Image
import torch, torchvision





class Stage1_Dataset:
    def __init__(self, imagefiles:tp.List[str], patchsize=512):
        self.imagefiles = imagefiles
        self.patchsize  = patchsize
    
    def __len__(self) -> int:
        return len(self.imagefiles)
    
    def __getitem__(self, i:int) -> torch.Tensor:
        imgf      = self.imagefiles[i]
        img       = PIL.Image.open(imgf).convert('RGB')
        #random crop
        y1,x1     = np.random.randint([self.patchsize]*2, img.size) #min:512, max:H/W
        y0,x0     = y1-self.patchsize, x1-self.patchsize
        patch     = img.crop([y0,x0,y1,x1])
        return torchvision.transforms.ToTensor()(patch)


class Stage2_Dataset:
    def __init__(self, matches:tp.List[tp.Dict], n=64, patchsize=512):
        self.n         = n
        self.patchsize = patchsize
        #filter items without matches
        self.matches   = [m for m in matches if len(m['points0']) > 1 ]
    
    def __len__(self):
        return len(self.matches)
    
    def __getitem__(self, i:int) -> (torch.Tensor, torch.Tensor, np.ndarray, np.ndarray):
        match     = self.matches[i]
        imgf0     = match['img0']
        imgf1     = match['img1']
        img0      = PIL.Image.open(imgf0).convert('RGB')
        img1      = PIL.Image.open(imgf1).convert('RGB')
        patchsize = self.patchsize
        
        points0, points1 = match['points0'], match['points1']
        #pick a random match and crop the images around it
        match_i   = np.random.randint(len(points0))
        y0,x0     = points0[match_i][::-1] - patchsize//2
        patch0    = img0.crop([y0,x0, y0+patchsize,x0+patchsize])
        
        y1,x1     = points1[match_i][::-1] - patchsize//2
        patch1    = img1.crop([y1,x1, y1+patchsize,x1+patchsize])
        
        #collect points that are inside both patches
        point_mask  = (points0 > (x0,y0)).all(1) & (points0 < (x0+patchsize,y0+patchsize)).all(1)
        point_mask &= (points1 > (x1,y1)).all(1) & (points1 < (x1+patchsize,y1+patchsize)).all(1)
        n           = self.n
        point_ixs   = np.random.permutation(np.argwhere(point_mask)[:,0])[:n]
        
        #convert from image-coordinates to patch-coordinates
        points0     = points0[point_ixs] - (x0,y0)
        points1     = points1[point_ixs] - (x1,y1)
        #append some out-of-range points if needed
        points0     = np.concatenate([points0, -1000*np.ones([n,2], dtype=points0.dtype)])[:n]
        points1     = np.concatenate([points1, -1000*np.ones([n,2], dtype=points1.dtype)])[:n]

        patch0      = torchvision.transforms.ToTensor()(patch0)
        patch1      = torchvision.transforms.ToTensor()(patch1)
        return patch0, patch1, points0, points1


def create_dataloader(dataset, batch_size, shuffle=False, num_workers='auto'):
    if num_workers == 'auto':
        num_workers = os.cpu_count()
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle, collate_fn=getattr(dataset, 'collate_fn', None),
                                       num_workers=num_workers, pin_memory=True,
                                       worker_init_fn=lambda x: np.random.seed(torch.randint(0,1000,(1,))[0].item()+x) )



def file_identifier(filename:str) -> tp.Union[str,None]:
    '''Extract the experiment/identifier from a file name
       e.g: CW_T001_L003_02.08.19_093548_022_CA.tiff -> CW_T001_L003
    '''
    return '_'.join(os.path.basename(filename).split('_')[:-4])

def file_date(filename:str) -> tp.Union[datetime.datetime,None]:
    '''Extract the date from the file name
       e.g: CW_T001_L003_02.08.19_093548_022_CA.tiff -> 2019.08.02
    '''
    date_candidates = filename.split('_')
    for datestring in date_candidates:
        splits      = datestring.split('.')
        try:
            assert len( [int(s) for s in splits] ) == 3
        except (ValueError, AssertionError):
            #does not look like a date
            continue
        
        (a,b,c) = splits
        if len(a) > 2:
            #interpreting as format YYYYMMDD
            [y,m,d] = map(int, [a,b,c])
        elif len(c) > 2:
            #interpreting as format DDMMYYYY
            [y,m,d] = map(int, [c,b,a])
        else:
            #interpreting as format DDMMYY
            [y,m,d] = map(int, [c,b,a])
            y       = (y+2000) if y<70 else (y+1900)    #1970-2069
        date = datetime.datetime(y, m, d);
        return date
        


def group_filenames(filenames:[str]) -> tp.Dict[str, tp.List[str]]:
    '''Sort filenames into groups with the same identifier'''
    filenames = [f for f in filenames if file_date(f) is not None]
    all_ids   = sorted( set(map(file_identifier, filenames)) )

    grouped_files = dict(
        [ (id,sorted([f for f in filenames if file_identifier(f)==id], key=file_date)) for id in all_ids]
    )
    #at least two images per sequence
    grouped_files = dict( [(k,v) for k,v in grouped_files.items() if len(v)>1] )
    return grouped_files
