'''
Generic structure for a dataset
Iterator based on code from pylearn2 (https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/datasets/dataset.py)
Pylearn2's license is as follows:
Copyright (c) 2011--2014, Universite de Montreal
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
__authors__ = "Markus Beissinger"
__copyright__ = "Copyright 2015, Vitruvian Science"
__credits__ = ["Markus Beissinger"]
__license__ = "Apache"
__maintainer__ = "OpenDeep"
__email__ = "dev@opendeep.com"

# standard libraries
import logging
import os
import shutil
# internal references
from opendeep.utils.file_ops import mkdir_p, get_file_type, download_file
import opendeep.utils.file_ops as files

log = logging.getLogger(__name__)

# variables for the dataset modes
SEQUENTIAL = 0
RANDOM     = 1


class Dataset(object):
    '''
    Default interface for a dataset object
    '''
    def __init__(self, filename=None, source=None, dataset_dir='../../datasets'):
        '''
        :param filename: string
        The name of the dataset file.
        :param source: string
        The URL or source for downloading the dataset.
        :param dataset_dir: string
        The directory relative to this file where the datasets should be stored.
        '''
        self.filename = filename
        self.source = source
        self.dataset_dir = dataset_dir
        self.file_type = None
        self.dataset_location = None

        self.install()


    def install(self):
        log.debug('Installing dataset %s', str(type(self)))
        # construct the actual path to the dataset
        prevdir = os.getcwd()
        os.chdir(os.path.split(os.path.realpath(__file__))[0])
        self.dataset_dir = os.path.realpath(self.dataset_dir)
        try:
            mkdir_p(self.dataset_dir)
            self.dataset_location = os.path.join(self.dataset_dir, self.filename)
        except:
            log.exception("Couldn't make the dataset path for %s with directory %s and filename %s",
                          str(type(self)),
                          self.dataset_dir,
                          str(self.filename))
            self.dataset_location = None
        finally:
            os.chdir(prevdir)

        # check if the dataset is already in the source, otherwise download it.
        # first check if the base filename exists - without all the extensions.
        # then, add each extension on and keep checking until the upper level, when you download from http.
        if self.dataset_location is not None:
            (dirs, fname) = os.path.split(self.dataset_location)
            split_fname = fname.split('.')
            accumulated_name = split_fname[0]
            found = False
            # first check if the filename was a directory (like for the midi datasets)
            if os.path.exists(os.path.join(dirs, accumulated_name)):
                found = True
                self.file_type = get_file_type(os.path.join(dirs, accumulated_name))
                self.dataset_location = os.path.join(dirs, accumulated_name)
                log.debug('Found file %s', self.dataset_location)
            # now go through the file extensions starting with the lowest level and check if the file exists
            if not found and len(split_fname)>1:
                for chunk in split_fname[1:]:
                    accumulated_name = '.'.join((accumulated_name, chunk))
                    self.file_type = get_file_type(os.path.join(dirs, accumulated_name))
                    if self.file_type is not None:
                        self.dataset_location = os.path.join(dirs, accumulated_name)
                        log.debug('Found file %s', self.dataset_location)
                        break

        # if the file wasn't found, download it.
        download_success = True
        if self.file_type is None:
            download_success = download_file(self.source, self.dataset_location)
            self.file_type = get_file_type(self.dataset_location)

        # if the file type is a zip, unzip it.
        unzip_success = True
        if self.file_type is files.ZIP:
            (dirs, fname) = os.path.split(self.dataset_location)
            post_unzip = os.path.join(dirs, '.'.join(fname.split('.')[0:-1]))
            unzip_success = files.unzip(self.dataset_location, post_unzip)
            # if the unzip was successful
            if unzip_success:
                # remove the zipfile and update the dataset location and file type
                log.debug('Removing file %s', self.dataset_location)
                os.remove(self.dataset_location)
                self.dataset_location = post_unzip
                self.file_type = get_file_type(self.dataset_location)
        if download_success and unzip_success:
            log.debug('Installation complete for %s. Yay!', str(type(self)))
        else:
            log.warning('Something went wrong installing dataset %s. Boo :(', str(type(self)))


    def uninstall(self):
        # TODO: Check if this shutil.rmtree is unsafe...
        log.debug('Uninstalling (removing) dataset %s for class %s...', self.dataset_location, str(type(self)))
        if self.dataset_location is not None and os.path.exists(self.dataset_location):
            # If we are trying to remove something not from the dataset directory, give a warning
            if not self.dataset_location.startswith(self.dataset_dir):
                log.critical("ATTEMPTING TO REMOVE A FILE NOT FROM THE DATASET DIRECTORY. LOCATION IS %s AND THE DATASET DIRECTORY IS %s",
                             self.dataset_location,
                             self.dataset_dir)
            shutil.rmtree(self.dataset_location)
        else:
            log.debug('%s\'s dataset_location was not valid. It was %s', str(type(self)), str(self.dataset_location))
        log.debug('Uninstallation (removal) successful for %s!', str(type(self)))


    def __iter__(self):
        return self.iterator()


    def iterator(self, mode=None, batch_size=None, minimum_batch_size=None, rng=None):
        '''
        :param mode: integer
        The type of iteration through the dataset. Can be SEQUENTIAL or RANDOM
        :param batch_size: integer
        The target batch size while iterating through the dataset
        :param minimum_batch_size: integer
        The minimum batch size allowed while iterating through the dataset (reached at the end)
        :param rng: random number generator object
        A random generator object with an interface such as numpy's random or theano's rng_mrg

        :return iterator: dataset.Iterator object
        An iterator object implementing the standard Python
        iterator protocol (i.e. it has an `__iter__` method that
        return the object itself, and a `next()` method that
        returns results until it raises `StopIteration`).
        '''
        log.critical("Iterator not implemented for dataset %s", str(type(self)))
        raise NotImplementedError()