import argparse


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    
    argparser.add_argument(
        '-m','--model',
        dest='model',
        metavar="M",
        default='None',
        help='The model to use. Select one of the following: Conv, dillution, lstm')
    args = argparser.parse_args()
    return args
