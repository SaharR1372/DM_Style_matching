The repository is based on [this repo](https://github.com/VICO-UoE/DatasetCondensation), please cite their paper [Dataset Condensation with Distribution Matching](https://arxiv.org/pdf/2110.04181) if you use the code. 
## Usage

```
python DM_GramMatching.py/DM_MeanStd_Matching.py  --dataset CIFAR10  --model ConvNet_gram  --ipc 10  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init real   --Iteration 20000 --num_exp 5  --num_eval 5  --save_path result_cifar10_DM_StyleMatching   --style_ratio 10000

python DM_KNearest.py  --dataset CIFAR10  --model ConvNet  --ipc 10  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init real   --Iteration 20000 --num_exp 5  --num_eval 5  --save_path result_cifar10_DM_KNearest --icd_ratio 10

```









