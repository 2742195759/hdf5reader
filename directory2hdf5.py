import h5py
import numpy as np
import os
import os.path as osp
import glob

def _directory2hdf5(root_group, dir_path):
    object_list = glob.glob(osp.join(dir_path, '*'))
    for obj in object_list:
        basename = osp.basename(obj)
        if osp.isfile(obj) :
            data = _read_file(obj, basename)
            if '.npy' in basename : 
                root_group.create_dataset(basename, data=data)
            #elif '.pkl' in basename:
            #    import pdb
            #    pdb.set_trace()
            #    root_group.attrs[basename] = data
        elif osp.isdir(obj):
            group = root_group.create_group(basename)
            _directory2hdf5(group, obj)

def _read_file(fullpath, basename):
    ret = None
    if '.pkl' in basename: 
        import pickle as pkl
        ret = pkl.load(open(fullpath, 'rb'))
    elif '.npy' in  basename:
        import numpy as np
        ret = np.load(fullpath)
    assert ret is not None, "Ret can't be None"
    return ret

def directory2hdf5(dir_path):
    """ 目前支持在dir_path中一种格式
        .npy 将作为数据
    """
    assert os.path.isdir(dir_path), "Input is not a directory path"
    upper_path = osp.dirname(dir_path)
    file_name = osp.basename(dir_path)
    output_file_name = osp.join(upper_path, file_name + '.hdf5')
    print ("Output file name", output_file_name)
    f = h5py.File(output_file_name, "w")
    _directory2hdf5(f, dir_path)
    f.close()

class DirectoryReader(object):
    def __init__(self):
        #assert os.path.isdir(dir_path), "Input is not a directory path"
        self.dirname2h5file = {}
        ...

    def findHdf5(self, fullpath):
        hdf5path = None
        suffix_path = []
        tmp = fullpath
        while '/' in tmp:
            dirname = osp.dirname(tmp)
            basename = osp.basename(tmp)
            hdf5path = osp.join(dirname, basename + '.hdf5')
            if osp.isfile(hdf5path): 
                hdf5path = osp.join(dirname, basename)
                break
            tmp = dirname
            suffix_path.append(basename)
        return hdf5path, suffix_path[::-1]

    def read(self, fullpath, to_numpy=True):
        assert '.npy' in fullpath, "Only Support .npy"
        hdf5path, suffixpath = self.findHdf5(fullpath)
        if hdf5path == None : 
            raise Exception("Hdf5 can't be found, Try np.load() ; but not implemented, contact xiongkun")
        if hdf5path not in self.dirname2h5file:
            print("Loading", hdf5path)
            f = h5py.File(hdf5path+'.hdf5', 'r')
            self.dirname2h5file[hdf5path] = f
        fp = self.dirname2h5file[hdf5path]

        ## read
        tmp_fp = fp
        for suf in suffixpath:
            tmp_fp = tmp_fp[suf]
        dataset = tmp_fp
        if to_numpy: return np.array(dataset)
        else: return dataset

import sys
if __name__ == "__main__":
    """
        Directory 2 Hdf5
    """
    print ("Start Converse:")
    dir_path = sys.argv[1]
    if dir_path[-1] == '/': dir_path = dir_path[:-1]
    print ("Input Path:", dir_path)
    if '.npy'  not in dir_path:
        print ("Convert Directory to HDF5")
        directory2hdf5(dir_path)
    else: 
        print ("Read npy file")
        reader = DirectoryReader()
        dataset = reader.read(dir_path)
        print (dataset.shape)

    # --------------------

    #dataset = reader.read('/home/xiongkun/Output/ProposalNet/referclef_val/features/ag9bKiPYX9y.npy')
    #print (dataset.shape)


    

