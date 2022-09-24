import os, glob, argparse, sys, itertools

from src import models, datasets, training

parser = argparse.ArgumentParser()
parser.add_argument('--inputfiles', help='glob-like path pattern to training images (e.g path/to/*.tiff)')
parser.add_argument('--segmentation_model', required=True)
#TODO: cuda

args = parser.parse_args()


if __name__ == '__main__':
    inputfiles = glob.glob(args.inputfiles)
    if len(inputfiles) == 0:
        print('Could not find any images')
        sys.exit(1)
    
    grouped_files = datasets.group_filenames(inputfiles)
    if len(grouped_files) == 0:
        print('Could not find any time series images')
        sys.exit(1)

    segnet = models.load_model(args.segmentation_model)
    simnet = models.RootTrackingModel()

    ds_stage1   = datasets.Stage1_Dataset(inputfiles)
    task_stage1 = training.Stage1_Task(simnet, seg_model=segnet)
    print('Training Stage 1')
    task_stage1.fit(datasets.create_dataloader(ds_stage1, 8))
    print()


    pairs       = [zip(group[1:], group[:-1]) for group in grouped_files.values()]
    pairs       = list(itertools.chain(*pairs))
    matches     = [models.full_inference(*imgpair, segnet, simnet) for imgpair in pairs]

    ds_stage2   = datasets.Stage2_Dataset(matches)
    task_stage2 = training.Stage2_Task(simnet)
    print('Training Stage 2')
    task_stage2.fit(datasets.create_dataloader(ds_stage2, 8))

    print( 'Saving to:', simnet.save('%Y-%m-%d_root_tracking') )

    print('Done')