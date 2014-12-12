import collections

class Logger(object):
    '''
    A simple logging class to print both to console and a log file.
    '''

    def __init__(self, outdir):
        self.logfile = outdir+"log.txt"
        # initialize as empty file
        with open(self.logfile,'w') as f:
            f.write('')
        
    # write to stdout and logfile without appending a newline
    def append(self, text):
        # if text is already a string
        if isinstance(text, basestring):
            pass
        # otherwise if it is an iterable
        elif isinstance(text, collections.Iterable):
            text = ' '.join([str(item) for item in text])
        # if not iterable and not already a string, just try to make a string out of it
        else:
            text = str(text)
            
        with open(self.logfile,'a') as f:
            f.write(text)
        print text,
            
    # write to stdout and logfile with appending a newline
    def log(self, text):
        # if text is already a string
        if isinstance(text, basestring):
            text = text+'\n'
        # otherwise if it is an iterable
        elif isinstance(text, collections.Iterable):
            text = ' '.join([str(item) for item in text])+'\n'
        # if not iterable and not already a string, just try to make a string out of it
        else:
            text = str(text)+'\n'
        
        with open(self.logfile,'a') as f:
            f.write(text)
        print text[:-1]