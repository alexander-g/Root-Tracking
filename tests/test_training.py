from src import training, models
import numpy as np
import torch



def test_sample_points():
    mock_segmaps = torch.zeros([2,1,64,64])
    mock_segmaps[1][0, 10:40, 10:40] = 0.75

    sampled_pts = training.sample_points(mock_segmaps, n=10)
    assert torch.stack(sampled_pts, 0).shape == (2,10,2)
    assert (sampled_pts[1] >= 10).all()
    assert (sampled_pts[1] <  40).all()


def test_stage1_task():
    model    = models.RootTrackingModel(c=4, backbone_pretrained=False)
    segmodel = models.UNet(c=4, backbone_pretrained=False)
    task     = training.Stage1_Task(model, epochs=2, seg_model=segmodel)

    batch     = torch.rand([2,3,128,128])
    loss,logs = task.training_step(batch)

def test_stage2_task():
    model    = models.RootTrackingModel(c=4, backbone_pretrained=False)
    task     = training.Stage2_Task(model, epochs=2)

    batch     = [
        torch.rand([2,3,128,128]),
        torch.rand([2,3,128,128]),
        torch.arange(20).reshape(2,-1, 2),
        torch.arange(20).reshape(2,-1, 2),
    ]
    loss,logs = task.training_step(batch)


def test_rotate_flip():
    x = torch.zeros([2,1,40,40])
    p = torch.as_tensor( [[1,1], [10,20]], ).reshape(2,2)
    x[(0,1), 0, p[:,0], p[:,1]] = 1
    
    x0,p0 = training.random_rotate_flip(x, p, tk=('rot', 0))
    assert (x0 == x).all()
    assert (p0 == p).all()

    x1,p1 = training.random_rotate_flip(x, p, tk=('rot', 1))
    assert tuple(np.argwhere(x1[0,0]).flatten()) in [(38,1)]
    assert tuple(np.argwhere(x1[1,0]).flatten()) in [(19,10)]
    assert tuple(p1[0]) in [(38,1)]
    assert tuple(p1[1]) in [(19,10)]
