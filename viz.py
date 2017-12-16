# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import numpy as np
import scipy.misc
from io import BytesIO as IO # Python 3.x
import pickle


class Logger(object):
    def __init__(self, log_file):
        self.log_file = log_file
        self.hist = {}

    def scalar_summary(self, tag, value):
        """Log a scalar variable."""
        data = self.hist.get(tag, [])
        data.append(value)
        self.hist[tag] = data
    
    def flush(self):
        outfile = open(self.log_file, "wb")
        pickle.dump(self.hist, outfile)
        outfile.close()
