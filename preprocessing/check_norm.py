
import numpy as np
import sys

def checking_npy(filename):
    t = np.load(filename)
    print (t.size)
    print (len(t))
    print (np.sum(t))
    #print (t)
    print (t.shape)
    print (t.max())


if __name__ == "__main__":
    argv = sys.argv 
    print (argv)
    filename = argv[1]
    checking_npy(filename)
