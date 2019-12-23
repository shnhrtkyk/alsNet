from argparse import ArgumentParser
import numpy as np
import csv
import glob
import os
import sys
import dataset
import glob

from argparse import ArgumentParser
from dataset import Dataset
import numpy as np
import os, sys
import logging
import importlib

# disable tensorflow debug information:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s [%(levelname)s]: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class pred:
    def __init__(self,in_files, density, kNN, thinFactor):
        self.in_files=in_files
        self.density =density
        self.kNN = kNN
        self.out_folder = ""
        self.thinFactor = thinFactor
        self.classnon = 3

        self.spacing = np.sqrt(kNN * self.thinFactor / (np.pi * density)) * np.sqrt(2) / 2 * 0.95  # 5% MARGIN
        print("Using a spacing of %.2f m" % self.spacing)


    def getpath(self):
        print ("path")

    def getknn(self):
        print ("knn")

    def pred(self):
        print ("pred")

    def mearge(self):
        print ("mearge")

    def save(self):
        print("save")

    def predictions(self):
        statlist = [["Filename", "StdDev_Classes", "Ground", "Lo Veg", "Hi Veg"]]
        for file_pattern in self.in_files:
            print(file_pattern)
            for file in glob.glob(file_pattern):
                outname = os.path.dirname(file) +  "/tmp/" + os.path.basename(file)[:-3]
                self.out_folder = os.path.dirname(file) + "/tmp/"
                if not os.path.exists(outname):
                    os.makedirs(outname)
                print("Loading file %s" % file)
                self.d = dataset.kNNBatchDataset(file=file, k=int(self.kNN * self.thinFactor), spacing=self.spacing)
                while True:
                    print("Processing batch %d/%d" % (self.d.currIdx, self.d.num_batches))
                    points_and_features, labels = self.d.getBatches(batch_size=1)
                    idx_to_use = np.random.choice(range(int(self.thinFactor * self.kNN)), self.kNN)
                    names = self.d.names
                    out_name = self.d.filename.replace('.la', '_c%04d.la' % self.d.currIdx)  # laz or las
                    out_path = os.path.join(self.out_folder, out_name)
                    if points_and_features is not None:
                        stats = dataset.ChunkedDataset.chunkStatistics(labels[0], 10)
                        print(stats)
                        rest = 1 - (stats['relative'][0] +
                                    stats['relative'][1] +
                                    stats['relative'][2] +
                                    stats['relative'][3] +
                                    stats['relative'][7] +
                                    stats['relative'][8])
                        perc = [stats['relative'][0],
                                stats['relative'][1],
                                stats['relative'][2],
                                stats['relative'][3],
                                stats['relative'][7],
                                stats['relative'][8],
                                rest]
                        stddev = np.std(perc) * 100
                        list_entry = [out_name, "%.3f" % stddev, *["%.3f" % p for p in perc]]
                        statlist.append(list_entry)
                        dataset.Dataset.Save(out_path, points_and_features[0][idx_to_use], names,
                                             labels=labels[0][idx_to_use], new_classes=None)
                    else:  # no more data
                        break

        with open(os.path.join(self.out_folder, "stats.csv"), "wb") as f:
            for line in statlist:
                f.write((",".join(line) + "\n").encode('utf-8'))



def main(in_files, density, kNN, thinFactor):
    mypred = pred(in_files, density, kNN, thinFactor)
    mypred.predictions()



def main_(in_files, density, kNN, out_folder, thinFactor):
    spacing = np.sqrt(kNN*thinFactor/(np.pi*density)) * np.sqrt(2)/2 * 0.95  # 5% MARGIN
    print("Using a spacing of %.2f m" % spacing)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    statlist = [["Filename", "StdDev_Classes", "Ground", "Lo Veg", "Hi Veg"]]
    for file_pattern in in_files:
        print(file_pattern)
        for file in glob.glob(file_pattern):
            print("Loading file %s" % file)

            d = dataset.kNNBatchDataset(file=file, k=int(kNN*thinFactor), spacing=spacing)
            while True:
                print("Processing batch %d/%d" % (d.currIdx, d.num_batches))
                points_and_features, labels = d.getBatches(batch_size=1)
                idx_to_use = np.random.choice(range(int(thinFactor*kNN)), kNN)
                names = d.names
                out_name = d.filename.replace('.la', '_c%04d.la' % d.currIdx)  # laz or las
                out_path = os.path.join(out_folder, out_name)
                if points_and_features is not None:
                    stats = dataset.ChunkedDataset.chunkStatistics(labels[0], 10)
                    print(stats)
                    stats = dataset.ChunkedDataset.chunkStatistics(labels[0], 10)
                    rest = 1 - (stats['relative'][0] +
                                stats['relative'][1] +
                                stats['relative'][2] +
                                stats['relative'][3] +
                                stats['relative'][7] +
                                stats['relative'][8])
                    perc = [stats['relative'][0],
                            stats['relative'][1],
                            stats['relative'][2],
                            stats['relative'][3],
                            stats['relative'][7],
                            stats['relative'][8],
                            rest]
                    stddev = np.std(perc) * 100
                    list_entry = [out_name, "%.3f" % stddev, *["%.3f" % p for p in perc]]
                    statlist.append(list_entry)
                    dataset.Dataset.Save(out_path, points_and_features[0][idx_to_use], names,
                                         labels=labels[0][idx_to_use], new_classes=None)
                else:  # no more data
                    break

    with open(os.path.join(out_folder, "stats.csv"), "wb") as f:
        for line in statlist:
            f.write((",".join(line) + "\n").encode('utf-8'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--inFiles',
                        default=[],
                        required=True,
                        help='input files (wildcard supported)',
                        action='append')
    parser.add_argument('--density', type=float, default=15, help='average point density')
    parser.add_argument('--kNN', default=2000000, type=int, required=False, help='how many points per batch [default: 200000]')
    parser.add_argument('--outFolder', required=False, help='where to write output files and statistics to')
    parser.add_argument('--thinFactor', type=float, default=1., help='factor to thin out points by (2=use half of the points)')
    args = parser.parse_args()

    main(args.inFiles, args.density, args.kNN, args.thinFactor)
