import os
import errno
from time import localtime, strftime
from typing import List, Callable, Optional

import torch
import numpy as np
from visdom import Visdom
from nest import register, Context


@register
def checkpoint(
    train_ctx: Context, 
    save_dir: str, 
    save_step: Optional[int] = None,
    save_on: Optional[List[str]] = None,
    save_final: bool = False,
    save_latest: bool = False,
    save_all: bool = False) -> None:
    """Checkpoint.
    """

    save_dir = os.path.abspath(save_dir)
    try:
        os.makedirs(save_dir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    # helper function
    def save_current_train_ctx(save_name):
        save_path = os.path.join(save_dir, save_name)
        if 'scheduler' in train_ctx and train_ctx.scheduler is not None:
            torch.save(dict(
                epoch_idx = train_ctx.epoch_idx + 1,
                batch_idx = train_ctx.batch_idx + 1,
                model = train_ctx.model.state_dict(),
                optimizer = train_ctx.optimizer.state_dict(),
                scheduler = train_ctx.scheduler.state_dict()), save_path)
        else:
            torch.save(dict(
                epoch_idx = train_ctx.epoch_idx + 1,
                batch_idx = train_ctx.batch_idx + 1,
                model = train_ctx.model.state_dict(),
                optimizer = train_ctx.optimizer.state_dict()), save_path)
        train_ctx.logger.info('checkpoint created at %s' % save_path)

    # checkpoint conditions
    if save_all:
        save_current_train_ctx(strftime("model_%Y_%m_%d_%H.%M.%S.pt", localtime()))
    if save_step is not None and (train_ctx.epoch_idx + 1) % save_step == 0:
        save_current_train_ctx('model_%d.pt' % train_ctx.epoch_idx)
    if save_final and (train_ctx.epoch_idx + 1) == train_ctx.max_epoch:
        save_current_train_ctx('model_final.pt')
    if save_latest:
        save_current_train_ctx('model_latest.pt')
    if save_on is not None:
        if not 'saved_metrics' in train_ctx:
            train_ctx.saved_metrics = train_ctx.metrics
        for condition in save_on:
            compare, metric = condition.split(' ',1)
            if compare == '>':
                if train_ctx.metrics[metric] >= train_ctx.saved_metrics[metric]:
                    save_current_train_ctx('model_best_%s.pt' % metric)
                    train_ctx.saved_metrics[metric] = train_ctx.metrics[metric]
            elif compare == '<':
                if train_ctx.metrics[metric] <= train_ctx.saved_metrics[metric]:
                    save_current_train_ctx('model_best_%s.pt' % metric)
                    train_ctx.saved_metrics[metric] = train_ctx.metrics[metric]

@register
def vis_trend(ctx: Context, train_ctx: Context, server: str, env: str, port: int = 80) -> None:
    """Track trend with Visdom.
    """

    if not 'vis' in ctx:
        ctx.vis = Visdom(server=server, port=port, env=env)
    
    try:
        for k, v in train_ctx.metrics.items():
            if isinstance(v, (int, float)):
                if ctx.vis.win_exists(k):
                    ctx.vis.line(
                        X = np.array([train_ctx.epoch_idx]),
                        Y = np.array([v]),
                        opts = dict(title=k, xlabel='epoch'),
                        win = k,
                        update = 'append')
                else:
                    ctx.vis.line(
                        X = np.array([train_ctx.epoch_idx]),
                        Y = np.array([v]),
                        opts = dict(title=k, xlabel='epoch'),
                        win = k)
        ctx.vis.save([env])
    except ConnectionError:
        train_ctx.logger.warning('Could not connect to visdom server "%s".' % server)


@register
def print_state(train_ctx: Context, formats: List[str], join_str: str = ' | ') -> None:
    """Print state.
    """

    def unescape(escapped_str):
        return bytes(escapped_str, "utf-8").decode("unicode_escape")

    def safe_format(format_str, **kwargs):
        try:
            return format_str.format(**kwargs)
        except:
            return None

    format_list = [safe_format(unescape(format_str), **vars(train_ctx)) for format_str in formats]
    output_str = unescape(join_str).join([val for val in format_list if val is not None])
    train_ctx.logger.info(output_str)


@register
def interval(
    train_ctx: Context, 
    hook: Callable[[Context], None], 
    epoch_interval: int = 1, 
    batch_interval: int = 1) -> None:
    """Skip interval.
    """

    if train_ctx.epoch_idx % epoch_interval == 0 and train_ctx.batch_idx % batch_interval == 0:
        hook(train_ctx)


@register
def update_lr(
    train_ctx: Context,
    epoch_step: int,
    factor: float = 0.1) -> None:
    """Update learning rate.
    """

    if (train_ctx.epoch_idx + 1) % epoch_step == 0:
        for idx, param in enumerate(train_ctx.optimizer.param_groups):
            param['lr'] = param['lr'] * factor
            train_ctx.logger.info(
                'LR of param group {:d} is updated to {:.4f}'.format(idx, param['lr']))
