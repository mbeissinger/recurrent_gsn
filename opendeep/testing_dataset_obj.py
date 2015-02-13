from opendeep.data.mnist import MNIST
import log.logger as logger
def main():
    logger.config_root_logger()
    m = MNIST()
    # m.uninstall()
    # m.install()
if __name__ == '__main__':
    main()