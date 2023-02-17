import os, glob, argparse, sys, itertools

from src import models, datasets, training

parser = argparse.ArgumentParser()
parser.add_argument('--inputfiles', help='glob-like path pattern to training images (e.g path/to/*.tiff)')
parser.add_argument('--segmentation_model', required=True)
parser.add_argument('--stage1_model', required=False, help='Re-use previously trained stage 1 model instead of starting of scratch')
parser.add_argument('--cuda', required=False, default='cpu', const='cuda', dest='device', action='store_const', help='Use CUDA')

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
    print(f'Training on {len(grouped_files)} sequences')



    segnet = models.load_model(args.segmentation_model).to(args.device)
    if args.stage1_model:
        print('Loading a previoulsy trained Stage 1 model')
        simnet = models.load_model(args.stage1_model).to(args.device)
    else:
        print('Training Stage 1')
        simnet = models.RootTrackingModel().to(args.device)

        ds_stage1   = datasets.Stage1_Dataset(inputfiles)
        task_stage1 = training.Stage1_Task(simnet, seg_model=segnet, lr=0.01)
        loader = datasets.create_dataloader(ds_stage1, batch_size=4)
        task_stage1.fit(loader, epochs=10)

        outputfile  = simnet.save('%Y-%m-%d_%Hh%Mm%Ss_root_tracking.stage1')
        print('Saved to:', outputfile )
        print()




    print('Matching image pairs for training stage 2 ...')
    pairs       = [zip(group[1:], group[:-1]) for group in grouped_files.values()]
    pairs       = list(itertools.chain(*pairs))
    matches     = []
    for i, imgpair in enumerate(pairs):
        print(f'{i:4d}/{len(pairs)}', end='\r')
        result = models.full_inference(*imgpair, segnet, simnet, n=50000, dev=args.device)
        if result['success']:
            matches.append(result)
    print(f'Successfully matched image pairs: {len(matches)}/{len(pairs)}')
    
    if len(matches) == 0:
        print('ERROR: Stage 1 could not match anything. Cannot proceeed.')
        sys.exit()
    




    print()
    print('Training Stage 2')

    ds_stage2   = datasets.Stage2_Dataset(matches)
    task_stage2 = training.Stage2_Task(simnet, lr=0.01).to(args.device)
    loader      = datasets.create_dataloader(ds_stage2, batch_size=4)
    task_stage2.fit(loader, epochs=10)

    outputfile  = simnet.save('%Y-%m-%d_%Hh%Mm%Ss_root_tracking.stage2')
    print('Saved to:', outputfile )

    print('Done')