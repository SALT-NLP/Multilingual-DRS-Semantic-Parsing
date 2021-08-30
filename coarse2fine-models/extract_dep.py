from __future__ import unicode_literals, print_function, division
from io import open
import argparse
import re

parser = argparse.ArgumentParser(description='Transform')
parser.add_argument('-infile', type=str, default='exp_data/en/gold/text.txt.conll')
parser.add_argument('-outfile', type=str, default='exp_data/en/gold/text.txt.seg.raw')
args = parser.parse_args()

dep=[]

with open(args.infile, 'r', encoding='utf-8') as reader, open(args.outfile, 'w', encoding='utf-8') as writer:
    for line in reader:
        for t in line.strip().split():
            if not t.split(u'￨')[1] in dep:
                dep.append(t.split(u'￨')[1])
    for d in dep:
        writer.write(d+' 1\n')
