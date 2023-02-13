from src import datasets
import tempfile, os
import numpy as np
import PIL.Image


def generate_s2_mock_data():
    mocks  = []
    tmpdir = tempfile.TemporaryDirectory(prefix='delete_me_')
    for i in range(4):
        mock = {}
        for j in range(2):
            rgb  = (np.random.random([128,128,3])*255).astype('uint8')
            path = os.path.join(tmpdir.name, f'{i}_{j}.jpg')
            PIL.Image.fromarray(rgb).save(path)
            mock[f'img{j}']    = path
            mock[f'points{j}'] = np.random.randint(0,128, size=[50*i, 2])  #0,50,100,150
        mocks.append(mock)
    return mocks, tmpdir



def test_stage2_ds():
    mock_matches, tmpdir = generate_s2_mock_data()

    ds = datasets.Stage2_Dataset(mock_matches, n=16, patchsize=64)

    #first mock contains no points, should be removed
    assert len(ds) == 3
    
    img0,img1, pt0, pt1 = ds[0]
    assert len(pt0) == len(pt1) == 16

    img0,img1, pt0, pt1 = ds[2]
    assert len(pt0) == len(pt1) == 16

    #make sure outputs are batch-able
    loader = datasets.create_dataloader(ds, batch_size=2, shuffle=True)
    batch  = next(iter(loader))
    assert batch[0].shape == (2,3,64,64)
    assert batch[2].shape == (2,16,2)


def test_filename_parsing():
    filenames = [
        'CW_T001_L003_02.08.19_093548_022_CA.tiff',
        'CW_T001_L003_06.08.18_174938_009_SS.tiff',
        'AD_T046_L003_25.07.18_121812_008_SS.tiff'
    ]

    assert datasets.file_identifier(filenames[0]) == 'CW_T001_L003'

    import datetime
    assert datasets.file_date(filenames[0]) == datetime.datetime(year=2019, month=8, day=2)

    groups = datasets.group_filenames(filenames)
    assert len(groups) == 1
    assert groups['CW_T001_L003'] == filenames[:2][::-1]


    filenames2 = [
        'project_T18_L0_2020.07.27_124630_30_mpa-B.tiff',
        'project_T18_L0_2021.06.27_124630_30_mpa-B.tiff'
    ]

    assert datasets.file_identifier(filenames2[0]) == 'project_T18_L0'
    assert datasets.file_date(filenames2[0])       == datetime.datetime(year=2020, month=7, day=27)

    groups = datasets.group_filenames(filenames2)
    assert len(groups) == 1



