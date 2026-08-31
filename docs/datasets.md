# Datasets

Everything lives under `data.data_path` (default `data/`), which is gitignored.

| `data.dataset` | resolution | classes | source |
| --- | --- | --- | --- |
| `MNIST` | 28×28×1 | 10 | torchvision, auto-download |
| `FashionMNIST` | 28×28×1 | 10 | torchvision, auto-download |
| `SVHN` | 32×32×3 | 10 | torchvision, auto-download |
| `CIFAR10` | 32×32×3 | 10 | torchvision, auto-download |
| `CIFAR100` | 32×32×3 | 100 | torchvision, auto-download |
| `TinyImageNet` | 64×64×3 | 200 | one preparation step, below |
| `ImageNet` | 128×128×3 | 1000 | CSV manifests, below |

All datasets are returned already normalised with their own channel statistics, and **no
augmentation is applied at load time** -- augmentation is DSA, applied inside the training
loops (`condense.dsa_strategy`, `eval.dsa_strategy`).

## Auto-downloaded datasets

Nothing to do. The first run downloads into `data.data_path`:

```bash
python train.py --config configs/ours/cifar10_ipc10.yaml
```

## TinyImageNet

Tiny ImageNet is not in torchvision, and decoding 110k JPEGs every run would dominate a
short job, so it is prepared once into a single tensor file:

```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P data
unzip -q data/tiny-imagenet-200.zip -d data
python scripts/prepare_tinyimagenet.py --root data/tiny-imagenet-200 --out data/tinyimagenet.pt
```

That writes `data/tinyimagenet.pt` holding uint8 images and int64 labels for the 100k train
and 10k val images, with classes ordered by `wnids.txt` so labels are stable across runs.
`get_dataset` then normalises and returns them.

At 64×64, `model.net_depth: 0` resolves to **ConvNetD4**, matching the DM literature. The
resolution is what triggers it, so this happens for any 64×64 dataset you add.

## ImageNet-1K

Read from CSV manifests rather than an ImageFolder tree, so a subset can be defined without
copying files. Expected layout:

```
data/imagenet/
    imagenet_train.csv      columns: image_id, label
    imagenet_val.csv        columns: image_id, label
    class_names.csv         one class name per line, in label order
    <the image files the manifests point at, paths relative to data/imagenet>
```

Images are resized and centre-cropped to 128×128. `image_id` may carry a leading separator;
it is stripped before joining.

### ImageNet subsets

The paper also reports ImageNette and ImageWoof, which are ten-class subsets of ImageNet-1K.
To build one, filter `imagenet_train.csv` / `imagenet_val.csv` to these ImageNet class
indices and renumber the labels to 0–9:

| subset | ImageNet class indices |
| --- | --- |
| ImageNette | 0, 217, 482, 491, 497, 566, 569, 571, 574, 701 |
| ImageWoof | 193, 182, 258, 162, 155, 167, 159, 273, 207, 229 |

Then point `data.data_path` at the directory holding the filtered manifests.

## Adding a dataset

One `elif` branch in `get_dataset` (`ddm/data.py`) returning

```python
channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader
```

Nothing else needs to change: the depth heuristic keys off `im_size`, and both the
condensation loop and every coreset selector are written against this tuple.
