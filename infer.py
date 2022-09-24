import os, argparse
import PIL.Image

from src import models


parser = argparse.ArgumentParser()
parser.add_argument('inputfile0')
parser.add_argument('inputfile1')
parser.add_argument('--segmentation_model', required=True)
parser.add_argument('--similarity_model',   required=True)

args = parser.parse_args()

if __name__=='__main__':
    segnet = models.load_model(args.segmentation_model)
    simnet = models.load_model(args.similarity_model)

    result = models.full_inference(args.inputfile0, args.inputfile1, segnet, simnet)

    destination = f'{os.path.basename(args.inputfile0)}.{os.path.basename(args.inputfile1)}.turnovermap.png'
    PIL.Image.fromarray(result['turnovermap']).save(destination)

    print('Done')
