# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re

SPACE_NORMALIZER = re.compile(r"\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def tokenize_line1(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    line = [token.split(u'￨')[0] for token in line.split()]
    return line

def tokenize_line2(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    line = [token.split(u'￨')[1] for token in line.split()]
    return line