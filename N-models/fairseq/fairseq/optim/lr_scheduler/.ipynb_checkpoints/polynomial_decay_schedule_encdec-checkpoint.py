# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('polynomial_decay_encdec')
class PolynomialDecaySchedule(FairseqLRScheduler):
    """Decay the LR on a fixed schedule."""

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)

        # set defaults
        args.warmup_updates = getattr(args, 'warmup_updates', 0) or 0

        self.lr_enc = args.lr_enc
        self.lr_dec = args.lr_dec
        self.end_learning_rate = args.end_learning_rate
        self.total_num_update_enc = args.total_num_update_enc
        self.total_num_update_dec = args.total_num_update_dec
        self.power = args.power
        if args.warmup_updates_enc > 0:
            self.warmup_factor_enc = 1. / args.warmup_updates_enc
        else:
            self.warmup_factor_enc = 1
        self.optimizer.set_lr_enc(self.warmup_factor_enc * self.lr_enc)

        if args.warmup_updates_dec > 0:
            self.warmup_factor_dec = 1. / args.warmup_updates_dec
        else:
            self.warmup_factor_dec = 1
        self.optimizer.set_lr_dec(self.warmup_factor_dec * self.lr_dec)


    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--force-anneal', '--fa', type=int, metavar='N',
                            help='force annealing at specified epoch')
        parser.add_argument('--warmup-updates-enc', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-updates-dec', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--end-learning-rate', default=0.0, type=float)
        
        parser.add_argument('--power', default=1.0, type=float)
        parser.add_argument('--total-num-update-enc', default=1000000, type=int)
        parser.add_argument('--total-num-update-dec', default=1000000, type=int)

    '''def get_next_lr(self, epoch):
        lrs = self.args.lr
        if self.args.force_anneal is None or epoch < self.args.force_anneal:
            # use fixed LR schedule
            next_lr = lrs[min(epoch, len(lrs) - 1)]
        else:
            # annneal based on lr_shrink
            next_lr = self.optimizer.get_lr()
        return next_lr'''

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        '''super().step(epoch, val_loss)
        self.lr = self.get_next_lr(epoch)
        self.optimizer.set_lr(self.warmup_factor * self.lr)
        return self.optimizer.get_lr()'''
        pass

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.args.warmup_updates_enc > 0 and num_updates <= self.args.warmup_updates_enc:
            self.warmup_factor_enc = num_updates / float(self.args.warmup_updates_enc)
            lr = self.warmup_factor_enc * self.lr_enc
        elif num_updates >= self.total_num_update_enc:
            lr = self.end_learning_rate
        else:
            warmup = self.args.warmup_updates_enc
            lr_range = self.lr_enc - self.end_learning_rate
            pct_remaining = 1 - (num_updates - warmup) / (self.total_num_update_enc - warmup)
            lr = lr_range * pct_remaining ** (self.power) + self.end_learning_rate
        self.optimizer.set_lr_enc(lr)

        if self.args.warmup_updates_dec > 0 and num_updates <= self.args.warmup_updates_dec:
            self.warmup_factor_dec = num_updates / float(self.args.warmup_updates)
            lr = self.warmup_factor_dec * self.lr_dec
        elif num_updates >= self.total_num_update_dec:
            lr = self.end_learning_rate
        else:
            warmup = self.args.warmup_updates_dec
            lr_range = self.lr_dec - self.end_learning_rate
            pct_remaining = 1 - (num_updates - warmup) / (self.total_num_update_dec - warmup)
            lr = lr_range * pct_remaining ** (self.power) + self.end_learning_rate
        self.optimizer.set_lr_dec(lr)

        return [self.optimizer.get_lr_enc(), self.optimizer.get_lr_dec()]
