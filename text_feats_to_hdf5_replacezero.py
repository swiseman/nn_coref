import numpy as np
import h5py
import multiprocessing as mp
import argparse

def firstpass_worker(task_q, results_q):
    """
    gets mention offsets
    """
    while True:
        d,line = task_q.get() # blocks
        if d == -1:
            return
        ment_starts = []
        fields = line.split("|")
        ments = int(fields[0])
        curr_idx = 2
        for i in xrange(1,ments):
            for j in xrange(i):
                feats = fields[curr_idx].split(" ")
                # index where next mention pair's features start
                ment_starts.append(len(feats))
                curr_idx += 1
            curr_idx += 1
        results_q.put((d,ment_starts))

def firstpass(fi,num_procs):
    """
    establishes document and mention offsets
    """
    task_q = mp.Queue(num_procs) # ensure can only put num_procs things on the q
    results_q = mp.Queue() # no max size
    
    # start worker procs
    for i in xrange(num_procs):
        mp.Process(target=firstpass_worker, args=(task_q,results_q)).start()
    
    # assign each line to a worker
    num_docs = 0
    with open(fi) as f:
        for i,line in enumerate(f):
            print "assigning doc", i
            task_q.put((i,line)) # blocks if no room in the Q
            num_docs += 1
    
    # done with the file, so tell workers they can die
    for i in xrange(num_procs):
        task_q.put((-1,None))
        
    # close the task q
    task_q.close()
    
    # collect the results
    ment_starts = [[] for i in xrange(num_docs)]
    print "merging results..."
    for i in xrange(num_docs):
        d,doc_ment_starts = results_q.get()
        ment_starts[d] = doc_ment_starts
    doc_starts = [0]
    [doc_starts.append(len(ment_starts[i])) for i in xrange(num_docs)]
    # now that ment starts are in order we can put back together
    all_ment_starts = [0]
    [all_ment_starts.extend(ment_starts[i]) for i in xrange(num_docs)]
    return np.cumsum(doc_starts).astype(np.int64), np.cumsum(all_ment_starts).astype(np.int64)


def doc_feats_worker(task_q, results_q):
    """
    concatenates all of a document's (binary) features
    """
    while True:
        d,line = task_q.get() # blocks
        if d == -1:
            return
        fields = line.split("|")
        ments = int(fields[0])
        doc_feats = []
        curr_idx = 2
        for i in xrange(1,ments):
            for j in xrange(i):
                feat_strs = fields[curr_idx].split(" ")
                # fill in the features starting at offset
                [doc_feats.append(int(fs)) for fs in feat_strs]
                curr_idx += 1
            curr_idx += 1
        results_q.put((d,np.array(doc_feats,dtype=np.int64)))

# this guy will actually make the features                
def merge_feats_worker(results_q, doc_starts, ment_starts, feat_pfx, replacement):
    """
    puts documents' features in correct order and saves matrix in hd5 format
    """
    num_docs = doc_starts.shape[0]-1
    feats = np.zeros(ment_starts[-1], dtype=np.int64)
    for i in xrange(num_docs):
        print "got doc", i, "merging..."
        d,nz = results_q.get()
        nz[nz == 0] = replacement
        feats[ment_starts[doc_starts[d]]:ment_starts[doc_starts[d+1]]] = nz
        del nz
    print "saving features in hdf5 format..."
    h5fi = h5py.File('%s-feats.h5' % (feat_pfx),'w')
    h5fi.create_dataset("feats",data=feats,compression="gzip")
    h5fi.close()
       

def secondpass(fi, num_procs, doc_starts, ment_starts, feat_pfx, replacement):
    task_q = mp.Queue(num_procs) # ensure can only put num_procs things on the q
    results_q = mp.Queue()
    
    # start worker procs
    for i in xrange(num_procs):
        mp.Process(target=doc_feats_worker, args=(task_q,results_q)).start()
    
    fbp = mp.Process(target=merge_feats_worker, args=(results_q,doc_starts,ment_starts, feat_pfx, replacement))
    fbp.start()
    
    # assign each line to a worker
    num_docs = 0
    with open(fi) as f:
        for i,line in enumerate(f):
            print "filling in doc", i
            task_q.put((i,line)) # blocks if no room in the Q
            num_docs += 1
    
    # done with the file, so tell workers they can die
    print "tryina kill procs..."
    for i in xrange(num_procs):
        task_q.put((-1,None))
    
    print "joining last guy..."
    fbp.join()
    task_q.close()
    results_q.close()


####################### for parsing anaphoricity feature files #################
def na_merge_feats_worker(results_q, doc_starts, ment_starts, feat_pfx):
    """
    puts documents' features in correct order and saves matrix in hd5 format
    """
    num_docs = doc_starts.shape[0]-1
    feats = np.zeros(ment_starts[-1], dtype=np.int32)
    for i in xrange(num_docs):
        print "got doc", i, "merging..."
        d,nz = results_q.get()
        feats[ment_starts[doc_starts[d]]:ment_starts[doc_starts[d+1]]] = nz
        del nz
    print "saving features in hdf5 format..."
    h5fi = h5py.File('%s-feats.h5' % (feat_pfx),'w')
    h5fi.create_dataset("feats",data=feats,compression="gzip")
    h5fi.close()


def na_firstpass_worker(task_q, results_q):
    """
    gets mention offsets
    """
    while True:
        d,line = task_q.get() # blocks
        if d == -1:
            return
        ment_starts = []
        fields = line.split("|")
        ments = int(fields[0])
        curr_idx = 1
        for i in xrange(ments):
            feats = fields[curr_idx].split(" ")            
            # index where next mention pair's features start
            ment_starts.append(len(feats))
            curr_idx += 1
        results_q.put((d,ment_starts))

def na_firstpass(fi,num_procs):
    """
    establishes document and mention offsets
    """
    task_q = mp.Queue(num_procs) # ensure can only put num_procs things on the q
    results_q = mp.Queue() # no max size
    
    # start worker procs
    for i in xrange(num_procs):
        mp.Process(target=na_firstpass_worker, args=(task_q,results_q)).start()
    
    # assign each line to a worker
    num_docs = 0
    with open(fi) as f:
        for i,line in enumerate(f):
            print "assigning doc", i
            task_q.put((i,line)) # blocks if no room in the Q
            num_docs += 1
    
    # done with the file, so tell workers they can die
    for i in xrange(num_procs):
        task_q.put((-1,None))
        
    # close the task q
    task_q.close()
    
    # collect the results
    ment_starts = [[] for i in xrange(num_docs)]
    print "merging results..."
    for i in xrange(num_docs):
        d,doc_ment_starts = results_q.get()
        ment_starts[d] = doc_ment_starts
    doc_starts = [0]
    [doc_starts.append(len(ment_starts[i])) for i in xrange(num_docs)]
    # now that ment starts are in order we can put back together
    all_ment_starts = [0]
    [all_ment_starts.extend(ment_starts[i]) for i in xrange(num_docs)]
    return np.cumsum(doc_starts).astype(np.int32), np.cumsum(all_ment_starts).astype(np.int32)


def na_doc_feats_worker(task_q, results_q, replacement):
    """
    concatenates all of a document's (binary) features
    """
    while True:
        d,line = task_q.get() # blocks
        if d == -1:
            return
        fields = line.split("|")
        ments = int(fields[0])
        doc_feats = []
        curr_idx = 1
        for i in xrange(ments):
            feat_strs = fields[curr_idx].split(" ")
            # fill in the features starting at offset
            [doc_feats.append(int(fs)) if fs != "0" else doc_feats.append(replacement) for fs in feat_strs]
            curr_idx += 1
        results_q.put((d,np.array(doc_feats,dtype=np.int32)))

    
def na_secondpass(fi, num_procs, doc_starts, ment_starts, feat_pfx, replacement):
    task_q = mp.Queue(num_procs) # ensure can only put num_procs things on the q
    results_q = mp.Queue()
    
    # start worker procs
    for i in xrange(num_procs):
        mp.Process(target=na_doc_feats_worker, args=(task_q,results_q,replacement)).start()
    
    fbp = mp.Process(target=na_merge_feats_worker, args=(results_q,doc_starts,ment_starts, feat_pfx))
    fbp.start()
    
    # assign each line to a worker
    num_docs = 0
    with open(fi) as f:
        for i,line in enumerate(f):
            print "filling in doc", i
            task_q.put((i,line)) # blocks if no room in the Q
            num_docs += 1
    
    # done with the file, so tell workers they can die
    print "tryina kill procs..."
    for i in xrange(num_procs):
        task_q.put((-1,None))
    
    print "joining last guy..."
    fbp.join()
    task_q.close()
    results_q.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_text_file", action="store", help="feature text file to convert to hdf5")
    parser.add_argument("pfx", action="store", help="prefix generated hdf5 files with this")
    parser.add_argument("mode", choices=("pw","ana"), action="store", help="pairwise or anaphoricity features in text file")
    parser.add_argument("-n","--num_procs", type=int, action="store", default=3, dest="num_procs", help="number of additional processes to spawn")
    parser.add_argument("-r","--replacement", type=int, action="store", default=-1, dest="replacement", help="replacement feature for zeros")
        
    args = parser.parse_args()
    if args.mode == "pw":
        feat_pfx = args.pfx + "-pw"
    else:
        feat_pfx = args.pfx + "-na"
        
    if args.mode == "pw":
        doc_starts, ment_starts = firstpass(args.feature_text_file, args.num_procs)
    else:
        doc_starts, ment_starts = na_firstpass(args.feature_text_file, args.num_procs)
    
    print "saving doc_starts and ment_starts in hdf5 format..."
    h5fi = h5py.File('%s-offsets.h5' % (feat_pfx),'w')
    h5fi.create_dataset("doc_starts",data=doc_starts)
    h5fi.create_dataset("ment_starts",data=ment_starts)
    h5fi.close()    

    print "will now fill in the", ment_starts[-1], "features..."
    if args.mode == "pw":
        # subtract 1 b/c we'll spawn an additional guy to consume worker output
        secondpass(args.feature_text_file, args.num_procs-1, doc_starts, ment_starts, feat_pfx, args.replacement)
    else:
        na_secondpass(args.feature_text_file, args.num_procs-1, doc_starts, ment_starts, feat_pfx, args.replacement)

if __name__ == "__main__":
    main()
