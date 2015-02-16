'''
@author: Markus Beissinger
University of Pennsylvania, 2014-2015
'''

import collections
import os, sys
import errno

class Logger(object):
    '''
    A simple logging class to print both to stdout and a log file at "outdir/log.txt".
    '''

    def __init__(self, outdir="."):
        # create the outdir if it doesn't exist
        mkdir(outdir)
        # add the trailing separator if it doesn't exist
        if outdir[-1] == "/":
            self.logfile = outdir+"log.txt"
        else:
            self.logfile = outdir+"/log.txt"
        # initialize log as empty file
        with open(self.logfile,'w') as f:
            f.write('')
        # set the self.out to stdout
        self.out = sys.stdout
        
    # write to stdout and logfile without appending a newline
    def append(self, text):
        # appropriately create the string from possible inputs
        text = self.parseText(text)
            
        with open(self.logfile,'a') as f:
            f.write(text)
        #print text,
        self.out.write(text)
            
    # write to stdout and logfile with appending a newline
    def log(self, text):
        # appropriately create the string from possible inputs
        text = self.parseText(text)
        
        with open(self.logfile,'a') as f:
            # need to add a newline at the end to make it work like 'print'
            f.write(text+'\n')
        #print text
        self.out.write(text+'\n')
        
    # parse some text to figure out if it is a collection (like the way print works)
    def parseText(self, text):
        # if text is already a string
        if isinstance(text, basestring):
            pass
        # otherwise if it is an iterable
        elif isinstance(text, collections.Iterable):
            text = ' '.join([str(item) for item in text])
        # if not iterable and not already a string, just try to make a string out of it
        else:
            text = str(text)
            
        return text
        
# Given the possibility of a logger instance, write to the logger. Otherwise, standard console print        
def maybeLog(logger, text):
    if logger is not None:
        logger.log(text)
    else:
        print text
        
# Given the possibility of a logger instance, append to the logger. Otherwise, standard console print
def maybeAppend(logger, text):
    if logger is not None:
        logger.append(text)
    else:
        print text,
        
#create a filesystem path if it doesn't exist.
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise