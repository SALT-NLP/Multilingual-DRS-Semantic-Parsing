# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq.data import LanguagePairDataset
from fairseq import utils

from .translation_dep import TranslationDepTask
from . import register_task


@register_task('translation_from_pretrained_mbert')
class TranslationFromPretrainedBERTTask(TranslationDepTask):
    """
    Translate from source language to target language with a model initialized with a multilingual pretrain.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationDepTask.add_args(parser)
        parser.add_argument('--prepend-bos', action='store_true',
                            help='prepend bos token to each sentence, which matches '
                                 'mBART pretraining')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict, src_dep_dict):
        super().__init__(args, src_dict, tgt_dict, src_dep_dict)
        for d in [src_dict, tgt_dict]:
            d.add_symbol('<mask>')
