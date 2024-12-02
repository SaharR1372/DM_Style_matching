# Official PyTorch implementation of "Decomposed Distribution Matching in Dataset Condensation", published as a conference paper at WACV 2025.

##Abstract

Dataset Condensation (DC) aims to reduce deep neural networks training efforts by synthesizing a small dataset such that it will be as effective as the original large dataset. Conventionally, DC relies on a costly bi-level optimization which prohibits its practicality. Recent research formulates DC as a distribution matching problem which circumvents the costly bi-level optimization. However, this efficiency sacrifices the DC performance.
   To investigate this performance degradation, we decomposed the dataset distribution into content and style. Our observations indicate two major shortcomings of: 1) style discrepancy between original and condensed data, and 2) limited intra-class diversity of condensed dataset.
   We present a simple yet effective method to match the style information between original and condensed data, employing statistical moments of feature maps as well-established style indicators.
   Moreover, we enhance the intra-class diversity by maximizing the Kullback–Leibler divergence within each synthetic class, \ie, content.
   We demonstrate the efficacy of our method through experiments on diverse datasets of varying size and resolution, achieving improvements of up to 8.3\% on CIFAR10, 7.9\% on CIFAR100, 3.6\% on TinyImageNet, 5\% on ImageNet-1K, 5.9\% on ImageWoof, 8.3\% on ImageNette, and 5.5\% in continual learning accuracy.




The repository is based on [this repo](https://github.com/VICO-UoE/DatasetCondensation), please cite their paper [Dataset Condensation with Distribution Matching](https://arxiv.org/pdf/2110.04181) if you use the code. 
## Usage

```
python DM_GramMatching.py/DM_MeanStd_Matching.py  --dataset CIFAR10  --model ConvNet_gram  --ipc 10  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init real   --Iteration 20000 --num_exp 5  --num_eval 5  --save_path result_cifar10_DM_StyleMatching   --style_ratio 10000

python DM_KNearest.py  --dataset CIFAR10  --model ConvNet  --ipc 10  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init real   --Iteration 20000 --num_exp 5  --num_eval 5  --save_path result_cifar10_DM_KNearest --icd_ratio 10

```









