from src import models
import numpy as np
import torch


def test_compute_descriptor_similarities():
    dsc0 = models.normalize( torch.rand([100, 32, 16, 16]) )
    dsc1 = models.normalize( torch.rand([200, 32, 16, 16]) )
    dsc0[0] = dsc1[0]
    
    sims = models.compute_descriptor_similarities(dsc0, dsc1)
    
    assert sims.shape == (100,200)
    assert 0 <= sims.max() <= 1
    assert 0 <= sims.min() <= 1
    assert sims[0,0] == 1

def test_compute_similarity_ratios():
    simmat = torch.as_tensor([
        [0.9, 0.95, 0.6],
    ])
    pt_all     = np.asarray( [ (10,10), (11,11), (100,100) ] )
    pt_matched = pt_all[:1]
    ratios = models.compute_similarity_ratios(simmat, pt_all, pt_matched, threshold=50)
    assert np.allclose(ratios, [0.95/0.6])

def test_sample_points_mixed():
    pt         = torch.randint(0,    2000, size=[3000,2])
    #add some points far away from the main set
    pt_extra   = torch.stack( [ torch.ones(10)*4000, torch.linspace(0,4000,10) ], -1 )
    pt         = torch.cat([pt, pt_extra])

    ixs_sampled = models.sample_points_mixed(pt, n_uniform=512, n_random=1000)
    pt_sampled  = pt[ixs_sampled]
    for p in pt_extra:
        #the far away points must be sampled
        assert p in pt_sampled


def test_bruteforce_match():
    dsc0 = models.normalize( torch.rand([2000, 32, 16,16]) )
    dsc1 = models.normalize( torch.rand([3000, 32, 16,16]) )
    pt0  = np.random.randint(0,4000, size=[2000,2])
    pt1  = np.random.randint(0,4000, size=[3000,2])

    i = np.random.randint(len(dsc0))
    j = np.random.randint(len(dsc1))
    dsc1[j] = dsc0[i]

    result = models.bruteforce_match(dsc0, dsc1, pt0, pt1, n=3000, ratio_threshold=1.0)
    assert (pt0[i], pt1[j]) in np.stack([result['points0'], result['points1']], 1)


def test_match_empty():
    seg0 = torch.zeros([200,200])
    seg1 = torch.rand([200,200])
    img0, img1 = [torch.zeros([3,64,64]) for i in range(2)]

    model   = models.RootTrackingModel(backbone_pretrained=False)
    result0 = model.match_images(img0, img1, seg0, seg1)
    
    assert len(result0['points0']) == len(result0['points1']) == 0
    assert result0['success'] == False

    result0['should_not_be_saved'] = '!'

    result1 = model.match_images(img0, img1, seg1, seg0)
    assert len(result1['points0']) == len(result1['points1']) == 0
    assert result1['success'] == False

    assert 'should_not_be_saved' not in result1

def test_compute_descriptors_at_points():
    img0    = np.random.random([3,100,100])
    pts     = np.random.randint(20,80, [5,2])

    c       = np.random.randint(3,30)
    d       = np.random.randint(3,10)
    model   = models.RootTrackingModel(c=c, backbone_pretrained=False)

    
    dsc     = model.compute_descriptors_at_points(img0, pts, box_size=10, dsc_size=d)
    assert dsc.shape == (len(pts), c, d, d)
    #assert ( ((dsc**2).sum(1)**0.5) == 1  ).all()      #not all ones because of interpolation
    assert ( ((dsc**2).sum(1)**0.5) <= 1.0001  ).all()  #can be slightly above 1 (probably rounding errors)


def test_create_turnovermap():
    seg0 = np.zeros([200,200], bool)
    seg1 = np.zeros([200,200], bool)

    seg0[50:100, 50:60]  = 1
    seg1[50:60,  50:100] = 1

    tmap = models.create_turnover_map(seg0, seg1)
    assert (tmap[50:60,  50:60] == 1).all()
    assert (tmap[60:100, 50:60] == 2).all()
    assert (tmap[50:60, 60:100] == 3).all()

    tmap_rgba = models.create_turnover_map_rgba(seg0, seg1)
    assert (tmap_rgba[50:60,  50:60] == 255).all()

def test_interpolation_map_dummy():
    imap = models.interpolation_map(np.zeros([1,2]), np.zeros([1,2]), (100,100))
    assert ( np.diff(imap, axis=0)[...,0] == 1).all()
    assert ( np.diff(imap, axis=1)[...,1] == 1).all()


def test_filter_points():
    p0 = np.random.randint(0,100, [1000,2])
    p1 = np.random.randint(0,100, [1000,2])

    #simulate a majority direction
    p0[:550] += 100000

    p0_, p1_ = models.filter_points(p0, p1, threshold=200)
    assert len(p0_) == len(p1_)
    assert 500 < len(p0_) < 1000
    assert all( [p in p0_ for p in p0[:500]] )

def test_match_images(model=None):
    if model is None:
        model   = models.RootTrackingModel(backbone_pretrained=False)
    
    img0, img1 = [torch.zeros([3,64,64]) for i in range(2)]
    seg0, seg1 = [torch.zeros([64,64])   for i in range(2)]
    seg0[10:20, 10:40]     = 1
    seg1[10:20, 10:40]     = 1
    result = model.match_images(img0, img1, seg0, seg1, n=10)
    assert len(result['points0'])  > 0


def test_save_model():
    model   = models.RootTrackingModel(backbone_pretrained=False)

    import tempfile,os
    tmpdir = tempfile.TemporaryDirectory(prefix='delete_me_')
    filename0 = model.save(os.path.join(tmpdir.name, '%Y-%m-%d_model'))
    assert os.path.exists(filename0)
    assert filename0.endswith('.pt.zip')
    assert '%Y' not in filename0

    model_reloaded = torch.package.PackageImporter(filename0).load_pickle('model', 'model.pkl')
    assert model_reloaded

    test_match_images(model_reloaded)

    #save again
    filename1 = model.save(os.path.join(tmpdir.name, '%Y-%m-%d_model2'))
    assert os.path.exists(filename1)

    tmpdir.cleanup()

