'''
Basic shared operations used for files
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.org"

# standard imports
import os
import errno
import urllib
import zipfile
import tarfile
import logging

log = logging.getLogger(__name__)

# variables for file format types
DIRECTORY = 0
ZIP       = 1
GZ        = 2
PKL       = 3
TAR       = 4
UNKNOWN   = 5


# create a filesystem path if it doesn't exist.
def mkdir_p(path):
    '''
    :param path: string
    Filesystem path to create
    '''
    path = os.path.realpath(path)
    log.debug('Attempting to make directory %s', path)
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            log.debug('Directory %s already exists!', path)
            pass
        else:
            log.exception('Error making directory %s', path)
            raise


# download a file from http to the destination file path
def download_file(url, destination):
    '''
    :param url: string
    The URL to download from
    :param destination: string
    The filesystem path (including file name) to download the file to

    :return: Boolean
    Whether or not it was successful
    '''
    destination = os.path.realpath(destination)
    log.debug('Downloading data from %s to %s', url, destination)
    try:
        page = urllib.urlopen(url)
        if page.getcode() is not 200:
            log.warning('Tried to download data from %s and got http response code %s', url, str(page.getcode()))
            return False
        urllib.urlretrieve(url, destination)
        return True
    except:
        log.exception('Error downloading data from %s to %s', url, destination)
        return False


# determine the type of file from its name
def get_file_type(file_path):
    '''
    :param file_path: string
    The filesystem path to the file

    :return: integer or None
    Returns the integer code to the file type defined in file_ops.py, or None if the file doesn't exist.
    '''
    file_path = os.path.realpath(file_path)
    if os.path.exists(file_path):
        # if it is a directory
        if os.path.isdir(file_path):
            return DIRECTORY
        # otherwise if it is a file
        elif os.path.isfile(file_path):
            (_, fname) = os.path.split(file_path)
            split_fname = fname.split('.')
            if len(split_fname) > 1:
                extension = split_fname[-1]
                if extension == 'zip':
                    return ZIP
                elif extension == 'gz':
                    return GZ
                elif extension == 'tar':
                    return TAR
                elif extension == 'pkl' or extension == 'p' or extension =='pickle':
                    return PKL
                else:
                    log.debug('Didn\'t recognize file extension %s for file %s', extension, file_path)
                    return UNKNOWN
            else:
                log.debug('File %s has no extension...', file_path)
                return UNKNOWN
        else:
            log.debug('File %s isn\'t a file or directory, but it exists... WHAT ARE YOU?!?', file_path)
            return UNKNOWN
    else:
        log.debug('File %s doesn\'t exist!', file_path)
        return None


# unzip a file to the destination directory
def unzip(source_filename, destination_dir='.'):
    '''
    :param source_filename: string
    Filesystem path to the file to unzip
    :param destination_dir: string
    Filesystem directory path for the file to unzip into

    :return: Boolean
    Whether or not it was successful
    '''
    source_filename = os.path.realpath(source_filename)
    destination_dir = os.path.realpath(destination_dir)
    log.debug('Unzipping data from %s to %s', source_filename, destination_dir)
    try:
        with zipfile.ZipFile(source_filename) as zf:
            zf.extractall(destination_dir)
            return True
    except:
        log.exception('Error unzipping data from %s to %s', source_filename, destination_dir)
        return False


# unzip a tarball to the destination directory
def untar(source_filename, destination_dir='.'):
    '''
    :param source_filename: string
    Filesystem path to the file to un-tar
    :param destination_dir: string
    Filesystem path for the file to un-tar into

    :return: Boolean
    Whether or not it was successful
    '''
    source_filename = os.path.realpath(source_filename)
    destination_dir = os.path.realpath(destination_dir)
    log.debug('Unzipping tarball data from %s to %s', source_filename, destination_dir)
    try:
        with tarfile.open(source_filename) as tar:
            tar.extractall(destination_dir)
            return True
    except:
        log.exception('Error unzipping tarball data from %s to %s', source_filename, destination_dir)
        return False

