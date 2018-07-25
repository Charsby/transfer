#**********************************
#The entrance of the whole process!

import sys
import getopt
from ConfigParser import ConfigParser
import nodule

def usage():
    print 'hello'
    
def main(config):
    nodule.full_process(config)

if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], 'h', ['help'])
    for name, value in opts:
        if name in ['-h', '--help']:
            usage()
            sys.exit()
    config = ConfigParser()
    config.read('./config.ini')
    main(config)