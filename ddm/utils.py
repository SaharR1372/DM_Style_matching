"""Small shared helpers: timing, logging, seeding and result bookkeeping."""
import json
import logging
import os
import random
import time

import numpy as np
import torch


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def format_time(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))


def set_seed(seed):
    """Seed python, numpy and torch together."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(save_path, filename='run.log'):
    """Log to stdout and to <save_path>/<filename>.

    Called once per entry point; re-configures the root logger so a second call inside the
    same process (a sweep, say) does not duplicate every line.
    """
    os.makedirs(save_path, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s | %(message)s',
        handlers=[logging.FileHandler(os.path.join(save_path, filename)),
                  logging.StreamHandler()])


def append_result(save_path, summary, filename='results.json'):
    """Append one run summary to <save_path>/results.json and return the file path.

    The file is a JSON list so a directory can accumulate the rows of an ablation table
    across separate invocations.
    """
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, filename)
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            try:
                rows = json.load(f)
            except json.JSONDecodeError:
                logging.warning('%s was not valid JSON; starting a new list', path)
    rows.append(summary)
    with open(path, 'w') as f:
        json.dump(rows, f, indent=2)
    return path


def save_image_grid(images, mean, std, path, nrow):
    """De-normalise and write a grid of condensed images for inspection."""
    from torchvision.utils import save_image
    vis = images.detach().cpu().clone()
    for ch in range(vis.shape[1]):
        vis[:, ch] = vis[:, ch] * std[ch] + mean[ch]
    save_image(vis.clamp(0, 1), path, nrow=nrow)
