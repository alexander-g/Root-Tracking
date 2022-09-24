import os, time, sys, copy
import numpy as np
import scipy
import torch, torchvision
from torchvision.models._utils import IntermediateLayerGetter
import PIL.Image
import skimage.morphology

#internal modules that should be included in a saved package
MODULES = []


class UNet(torch.nn.Module):
    '''Backboned U-Net'''

    class UpBlock(torch.nn.Module):
        def __init__(self, in_c, out_c, inter_c=None):
            torch.nn.Module.__init__(self)
            inter_c        = inter_c or out_c
            self.conv1x1   = torch.nn.Conv2d(in_c, inter_c, 1)
            self.convblock = torch.nn.Sequential(
                torch.nn.Conv2d(inter_c, out_c, 3, padding=1, bias=0),
                torch.nn.BatchNorm2d(out_c),
                #torch.nn.ReLU(),  #relu applied conditionally in forward()
            )
        def forward(self, x, skip_x, relu=True):
            x = torch.nn.functional.interpolate(x, skip_x.shape[2:])
            x = torch.cat([x, skip_x], dim=1)
            x = self.conv1x1(x)
            x = self.convblock(x)
            x = torch.relu(x) if relu else x
            return x
    
    def __init__(self, c=32, backbone_pretrained=True):
        torch.nn.Module.__init__(self)
        return_layers = dict(relu='out0', layer1='out1', layer2='out2', layer3='out3', layer4='out4')
        resnet        = torchvision.models.resnet18(backbone_pretrained, progress=False)
        self.backbone = IntermediateLayerGetter(resnet, return_layers )
        
        C = 512  #resnet18
        self.up0 = self.UpBlock(C    + C//2,  C//2)
        self.up1 = self.UpBlock(C//2 + C//4,  C//4)
        self.up2 = self.UpBlock(C//4 + C//8,  C//8)
        self.up3 = self.UpBlock(C//8 + C//8,   64)
        self.up4 = self.UpBlock(  64 +    3,   c)
        self.cls = torch.nn.Conv2d(c, 1, 3, padding=1)
    
    def forward(self, x, return_features=False):
        x = x.to(list(self.parameters())[0].device)
        X = self.backbone(x)
        X = ([x] + [X[f'out{i}'] for i in range(5)])[::-1]
        x = X.pop(0)
        x = self.up0(x, X[0])
        x = self.up1(x, X[1])
        x = self.up2(x, X[2])
        x = self.up3(x, X[3])
        x = self.up4(x, X[4], relu=not return_features)
        if return_features:
            return normalize(x)
        x0 = torch.sigmoid(self.cls(x))
        return x0
    
    @staticmethod
    def load_image(path):
        return PIL.Image.open(path) / np.float32(255)
    
    def process_image(self, image, progress_callback='ignored', threshold=0.5):
        if isinstance(image, str):
            image = self.load_image(image)
        
        x = torchvision.transforms.ToTensor()(image)[None]
        with torch.no_grad():
            y = self(x).cpu()
        output = y.numpy().squeeze()
        if threshold is not None:
            output = (output > threshold)*1
        return output
    
    def save(self, destination):
        if isinstance(destination, str):
            destination = time.strftime(destination)
            if not destination.endswith('.pt.zip'):
                destination += '.pt.zip'
        try:
            import torch_package_importer as imp
            #re-export
            importer = (imp, torch.package.sys_importer)
        except ImportError as e:
            #first export
            importer = (torch.package.sys_importer,)
        with torch.package.PackageExporter(destination, importer) as pe:
            #interns = [__name__.split('.')[-1]]+MODULES
            interns = [__name__]+MODULES

            pe.intern(interns)
            pe.intern(['torchvision.models.resnet.**'])
            pe.extern('**', exclude=['torchvision.**'])
            externs = ['torchvision.ops.**', 'torchvision.datasets.**', 'torchvision.io.**', 'torchvision.models.*']
            pe.intern('torchvision.**', exclude=externs)
            pe.extern(externs)
            
            #force inclusion of internal modules + re-save if importlib.reload'ed
            for inmod in interns:
                if inmod in sys.modules:
                    pe.save_source_file(inmod, sys.modules[inmod].__file__, dependencies=True)
                else:
                    pe.save_source_string(inmod, importer[0].get_source(inmod))
            
            self = self.cpu().eval().requires_grad_(False)
            pe.save_pickle('model', 'model.pkl', self)
        return destination
    

def load_model(file_path:str) -> torch.nn.Module:
    '''Load a saved model'''
    return torch.package.PackageImporter(file_path).load_pickle('model', 'model.pkl')


def normalize(x, axis=-3):
    denom = (x**2).sum(axis, keepdims=True)**0.5
    return x / denom


def extract_descriptors(x, pts_yx:list, box_size=64, dsc_size=16):
    '''Extract patches from featuremap x at locations pts_yx'''
    pts_xy = [torch.flip(yx, dims=[1]) for yx in pts_yx]
    boxes  = [torch.cat([xy-box_size//2, xy+box_size//2,], -1).float() for xy in pts_xy]
    dsc    = torchvision.ops.roi_align(x, boxes, dsc_size, sampling_ratio=1)
    return dsc


def sample_points_mixed(points, n_uniform, n_random):
    '''Samples up to `n_uniform` spatially uniform `points` and `n_random` points randomly'''
    points       = np.asarray(points)
    p_min, p_max = points.min(0), points.max(0)
    _n     = int(np.ceil(n_uniform**0.5))
    grid   = np.stack(np.meshgrid(*np.linspace(p_min, p_max, _n).T, indexing='ij'), -1).reshape(-1,2)
    dists  = ((grid[:,None] - points[None])**2).sum(-1)
    result = dists.argmin(1)
    result = np.concatenate([result, np.random.permutation(len(points))[:n_random]])
    result = np.unique(result)
    return result


def filter_points(p0, p1, threshold=50):
    delta      = (p1-p0)
    median     = np.median(delta, axis=0)  #TODO: median not optimal
    deviation  = ((delta - median)**2).sum(-1)**0.5
    ok         = (deviation < threshold)
    return p0[ok], p1[ok]


def interpolation_map(p0,p1, shape):
    '''Creates a map with coordinates from image1 to image0 according to matched points p0 and p1'''
    #direction vectors from each point1 to corresponding point0
    delta = (p1 - p0).astype('float32')
    
    #additional corner points, for extraploation
    cpts  = np.array([(0,0), (0,shape[1]), (shape[0],0), shape])
    #get their values via nearest neighbor
    delta_corners = scipy.interpolate.NearestNDInterpolator(p0, delta)(*cpts.T)
    #add them to the pool of known points
    p0    = np.concatenate([p0, cpts])
    delta = np.concatenate([delta, delta_corners])
    
    #densify the set of sparse points
    Y,X   = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    delta_linear = scipy.interpolate.LinearNDInterpolator(p0, delta)(Y,X)
    
    #convert from direction vectors to coordinates again
    return delta_linear + np.stack([Y,X], axis=-1)

def warp(seg, imap):
    return scipy.ndimage.map_coordinates(seg, imap.transpose(2,0,1), order=1)

def create_turnover_map(seg0, seg1):
    gmap = np.zeros( seg0.shape[:2], 'int8' )
    seg0 = np.asarray(seg0>0.5)
    seg1 = np.asarray(seg1>0.5)
    isec = seg0 & seg1
    gmap[isec]         = 1
    gmap[seg0 * ~isec] = 2
    gmap[seg1 * ~isec] = 3
    return gmap

def create_turnover_map_rgba(seg0, seg1):
    gmap      = create_turnover_map(seg0, seg1)
    gmap_rgba = np.zeros( seg0.shape[:2]+(4,), 'uint8' )
    gmap_rgba[gmap==0] = ( 39, 54, 59,  0)
    gmap_rgba[gmap==1] = (255,255,255,255)
    gmap_rgba[gmap==2] = (226,106,116,255)
    gmap_rgba[gmap==3] = ( 96,209,130,255)
    return gmap_rgba

#TODO: skeletonized turnover



class RootTrackingModel(UNet):
    interpolation_map       = staticmethod(interpolation_map)
    warp                    = staticmethod(warp)
    create_turnover_map_rgba  = staticmethod(create_turnover_map_rgba)

    def match_images(self, img0, img1, seg0, seg1, n=5000, ratio_threshold=1.1, cyclic_threshold=4, dev='cpu'):
        assert len(img0.shape) == len(img1.shape) == 3
        assert len(seg0.shape) == len(seg1.shape) == 2

        img0       = torchvision.transforms.ToTensor()(img0) if not torch.is_tensor(img0) else img0
        img1       = torchvision.transforms.ToTensor()(img1) if not torch.is_tensor(img1) else img1

        skl0       = skimage.morphology.skeletonize(np.asarray(seg0)>0.5)
        skl1       = skimage.morphology.skeletonize(np.asarray(seg1)>0.5)
        skl_p0     = np.argwhere(skl0)
        skl_p1     = np.argwhere(skl1)
        
        if len(skl_p1)==0 or len(skl_p0)==0:
            #no roots found, nothing to match
            return copy.deepcopy(EMPTY_RESULT)
        
        c      = 16
        with torch.no_grad():
            dsc0   = self.compute_descriptors_at_points(img0, skl_p0, dev, box_size=64, dsc_size=c)
            dsc1   = self.compute_descriptors_at_points(img1, skl_p1, dev, box_size=64, dsc_size=c)
        self.cpu(); torch.cuda.empty_cache() if dev=='cuda' else None;

        result = bruteforce_match(dsc0, dsc1, skl_p0, skl_p1, n, 512, ratio_threshold, cyclic_threshold)
        result['matched_percentage'] = len(result['points0']) / np.int32(len(skl_p0))
        p0,p1 = filter_points(result['points0'], result['points1'])
        result['points0'] = p0
        result['points1'] = p1
        #TODO: success
        return result
    
    def compute_descriptors_at_points(self, img, points, dev='cpu', **kw):
        img    = torch.as_tensor(img, dtype=torch.float32)
        points = torch.as_tensor(points)

        ft_map = self.to(dev)( img[None].to(dev), return_features=True )[0].cpu()
        dtype  = torch.float16 if dev=='cuda' else torch.float32
        ft_dsc = extract_descriptors(ft_map[None], [points], **kw).to(dtype).to(dev)
        return ft_dsc
    
    #legacy
    bruteforce_match       = lambda self, *a,**kw: self.match_images(*a[:4], **kw)
    create_growth_map_rgba = create_turnover_map_rgba
    

EMPTY_RESULT = {
    'points0':np.array([], 'int16').reshape(-1,2),
    'points1':np.array([], 'int16').reshape(-1,2),
    'scores' :np.array([], 'float32'),
    'ratios' :np.array([], 'float32'),
    'matched_percentage': 0,
    'success': False,
}


def compute_descriptor_similarities(dsc0:torch.Tensor, dsc1:torch.Tensor) -> torch.Tensor:
    c    = dsc0.shape[2]
    sims = torch.einsum('nchw,mchw->nm', dsc0, dsc1).cpu().float() / c**2
    #scale from -1..+1 to 0..+1
    sims = sims/2 + 0.5
    return sims

def compute_similarity_ratios(sim_matrix:torch.Tensor, pt_all:np.ndarray, pt_matched:np.ndarray, threshold=64):
    sim_matched     = sim_matrix.max(-1)[0]
    #set dists within 64px of pt_matched to zero to find second largest peak
    pt_distances    = np.abs(pt_all[None] - pt_matched[:,None]).max(-1)
    sim_matrix_     = sim_matrix.clone().numpy()
    sim_matrix_[pt_distances < threshold] = 0
    sim_reverse     = sim_matrix_.max(-1)
    ratios          = sim_matched / sim_reverse
    return ratios

def compute_cyclic_distances(ft1_matched, ft0, pt0_all, pt0_batch):
    sim_mat_cyclic  = compute_descriptor_similarities(ft1_matched, ft0)
    pt0_cyclic      = pt0_all[sim_mat_cyclic.argmax(-1)]
    dists_cyclic    = np.sum((pt0_cyclic - pt0_batch)**2, axis=-1)**0.5
    return dists_cyclic

def bruteforce_match(
    ft0:torch.Tensor, 
    ft1:torch.Tensor, 
    pt0:np.ndarray, 
    pt1:np.ndarray, 
    n:int, 
    step:int               = 512, 
    ratio_threshold:float  = 1.1, 
    cyclic_threshold:float = 4
):
    assert len(ft0) == len(pt0)
    assert len(ft1) == len(pt1)

    n      = min(n, len(ft0), len(ft1))
    ixs    = sample_points_mixed(pt0, n_uniform=512, n_random=n)[:n]
    result = copy.deepcopy(EMPTY_RESULT)

    #process in batches to save memory
    for i in range(0, n-1, step):
        ixs_batch       = ixs[i:][:step]
        ft0_batch       = ft0[ixs_batch]
        pt0_batch       = pt0[ixs_batch]

        #compute similarity matrix
        sim_matrix      = compute_descriptor_similarities(ft0_batch, ft1)
        #best matches
        ixs_matched     = sim_matrix.argmax(-1)
        pt1_matched     = pt1[ixs_matched]
        sim_matched     = sim_matrix.max(-1)[0]

        #compare the best similarities to the second best ones
        sim_ratios      = compute_similarity_ratios(sim_matrix, pt1, pt1_matched)
        ratios_ok       = (sim_ratios > ratio_threshold)
        
        #cycle consistency: matches should match back to the same coordinates (within a threshold)
        cyclic_dists    = compute_cyclic_distances(ft1[ixs_matched], ft0, pt0, pt0_batch)
        cyclic_ok       = (cyclic_dists < cyclic_threshold)

        #accept only matches that pass both tests
        all_ok          = np.array(ratios_ok & cyclic_ok).astype(bool)

        result['points0'] = np.concatenate([result['points0'], pt0_batch[all_ok]]).astype('int16')
        result['points1'] = np.concatenate([result['points1'], pt1_matched[all_ok]]).astype('int16')
        result['scores']  = np.concatenate([result['scores'],  sim_matched[all_ok]])
        result['ratios']  = np.concatenate([result['ratios'],  sim_ratios[all_ok]])

    return result


def full_inference(imagefile0:str, imagefile1:str, segmentation_model:UNet, similarity_model:RootTrackingModel) -> dict:
    img0 = UNet.load_image(imagefile0)
    img1 = UNet.load_image(imagefile1)

    seg0 = segmentation_model.process_image(img0)
    seg1 = segmentation_model.process_image(img1)

    result         = similarity_model.match_images(img0, img1, seg0, seg1)
    result['img0'] = imagefile0
    result['img1'] = imagefile1

    #TODO: if success:

    imap           = interpolation_map(result['points1'], result['points0'], seg0.shape)
    warped_seg0    = warp(seg0, imap)
    tmap           = create_turnover_map_rgba( warped_seg0>0.5, seg1>0.5, )
    result['turnovermap'] = tmap[..., :3]

    return result


