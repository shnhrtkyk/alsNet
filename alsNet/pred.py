from argparse import ArgumentParser
import numpy as np
import csv
import glob
import os
import sys
import dataset
import glob

from alsNetRefactored import AlsNetContainer
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



def test(in_files, density, kNN, thinFactor):
    mypred = pred(in_files, density, kNN, thinFactor)
    mypred.predictions()



def main(in_files, density, kNN, out_folder, thinFactor):
    spacing = np.sqrt(kNN*thinFactor/(np.pi*density)) * np.sqrt(2)/2 * 0.95  # 5% MARGIN
    print("Using a spacing of %.2f m" % spacing)
    print(out_folder)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    # load trained  model
    print("Loading trained model")

    model = AlsNetContainer(num_feat=3, num_classes=3, num_points=2000000, output_base=out_folder, arch="")
    model.load_model("/mnt/ssd/shino/log_tm/models/alsNet.ckpt")

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
                # tmp resilt dir
                out_name_tmp =  os.path.splitext(os.path.basename(out_name))[0]
                out_folder_tmp = out_folder + out_name_tmp

                print(out_folder_tmp)

                if not os.path.exists(out_folder_tmp):
                    os.makedirs(out_folder_tmp)

                out_path = os.path.join(out_folder_tmp, out_name)
                print(out_path)

                if points_and_features is not None:
                    # pred
                    acc = model.test_single(points_and_features[0][idx_to_use],
                                            save_to=out_path,
                                            save_prob=False, unload=True)


                    # dataset.Dataset.Save(out_path, points_and_features[0][idx_to_use], names,
                    #                      labels=labels[0][idx_to_use], new_classes=None)
                else:  # no more data
                    break

            # finish prediction one las file
            # merging result
            print ("Loading reference dataset")
            ref_ds = Dataset(file)
            ref_points = ref_ds._xyz
            out_labels = ref_ds.labels
            prob_sums = np.zeros((ref_points.shape[0], MAX_CLASSES))
            prob_counts = np.zeros((ref_points.shape[0],))
            print("Building 2D kD-Tree on the reference dataset")
            tree = ckdtree.cKDTree(ref_points[:, 0:2])  # only on 2D :D
            #get predicted las files
            input_files = os.listdir(out_folder_tmp)
            for filepattern in in_files:
                for file in glob.glob(filepattern):
                    input_files.append(file)

            for fileidx, file in enumerate(input_files):
                print("Processing file %d" % fileidx)
                ds = Dataset(file)
                points = np.hstack((ds.points_and_features, np.expand_dims(ds.labels, -1)))
                names = ds.names
                prob_ids_here = []
                prob_ids_ref = []
                for idx, name in enumerate(names):
                    if name.startswith('prob_class'):
                        prob_ids_here.append(idx + 3)
                        prob_ids_ref.append(int(name.split('prob_class')[-1]))

                for ptidx in range(points.shape[0]):
                    xy = points[ptidx, 0:2]
                    ref_ids = tree.query_ball_point(xy, r=0.0001, eps=0.0001)
                    if len(ref_ids) > 1:
                        ref_id = ref_ids[np.argmin(np.abs(ref_points[ref_ids, -1] - points[ptidx, 3]), axis=0)]
                    elif len(ref_ids) == 0:
                        print("Point not found: %s" % xy)
                        continue
                    else:
                        ref_id = ref_ids[0]
                    prob_counts[ref_id] += 1
                    probs_here = points[ptidx, prob_ids_here]
                    prob_sums[ref_id, prob_ids_ref] += probs_here
                del ds
                del points

            # clear memory
            ref_ds = None

            out_points = ref_points
            print(prob_counts)
            print(prob_sums[ref_id, :])

            prob_avgs = prob_sums / prob_counts[:, np.newaxis]
            print(prob_avgs)
            print(prob_avgs[ref_id, :])
            new_max_class = np.zeros((ref_points.shape[0]))
            for i in range(ref_points.shape[0]):
                curr_point = prob_sums[i, :] / prob_counts[i]
                curr_point_max = np.argmax(curr_point)
                new_max_class[i] = curr_point_max

            final = np.zeros((ref_points.shape[0], 4))
            final[:, :3] = ref_points[:, :3]
            new_max_class = np.where(new_max_class == 2, 6, new_max_class)
            new_max_class = np.where(new_max_class == 0, 2, new_max_class)
            new_max_class = np.where(new_max_class == 1, 6, new_max_class)
            final[:, 3] = new_max_class
            # save mearged data
            out_name_fin = os.path.splitext(os.path.basename(out_name))[0]
            out_folder_fin = out_folder + "_" + out_name_fin
            if not os.path.exists(out_folder_fin):
                os.makedirs(out_folder_fin)

            savename = out_name_fin + ".txt"

            np.savetxt(savename, final)







if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--inFiles',
                        default=[],
                        required=True,
                        help='input files (wildcard supported)',
                        action='append')
    parser.add_argument('--density', type=float, default=15, help='average point density')
    parser.add_argument('--kNN', default=200000, type=int, required=False, help='how many points per batch [default: 200000]')
    parser.add_argument('--outFolder', required=False, help='where to write output files and statistics to')
    parser.add_argument('--thinFactor', type=float, default=1., help='factor to thin out points by (2=use half of the points)')
    args = parser.parse_args()

    out_folder = "/mnt/ssd/shino/tm/"


    main(args.inFiles, args.density, args.kNN, out_folder , args.thinFactor)
