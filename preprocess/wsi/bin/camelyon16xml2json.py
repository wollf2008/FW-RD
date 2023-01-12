import sys
import os
import argparse
import logging


sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

from wsi.data.annotation import Formatter  # noqa

parser = argparse.ArgumentParser(description='Convert Camelyon16 xml format to'
                                 'internal json format')
parser.add_argument('xml_path', default=None, metavar='XML_PATH', type=str,
                    help='Path to the input Camelyon16 xml annotation file')
parser.add_argument('json_path', default=None, metavar='JSON_PATH', type=str,
                    help='Path to the output annotation in json format')


def run(args):
    xml_paths = os.listdir(args.xml_path)
    for i in xml_paths:
        xml_name = i
        jason_name = i.replace('.xml','.json')
        Formatter.camelyon16xml2json(os.path.join(args.xml_path, xml_name), os.path.join(args.json_path, jason_name))


def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
