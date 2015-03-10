import unittest
import logging
import opendeep.log.logger as logger
from opendeep.models.multi_layer.convolutional_network import AlexNet

class TestAlexNet(unittest.TestCase):

    def setUp(self):
        # configure the root logger
        logger.config_root_logger()
        # get a logger for this session
        self.log = logging.getLogger(__name__)

    def testInitCudaconv(self):
        self.log.info("creating alexnet with lib_conv=cudaconvnet")
        AlexNet(lib_conv='cudaconvnet')

    def testInitCUDNN(self):
        self.log.info("creating alexnet with lib_conv=cudnn")
        AlexNet(lib_conv='cudnn')

    def tearDown(self):
        pass
