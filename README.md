# MICCAI2020-Midline

Context-Aware Refinement Network Incorporating Morphology Prior for Delineation of Largely Deformed Brain Midline (MICCAI 2020)

[paper](https://arxiv.org/abs/2007.05393?context=cs)

## Install

Install all dependent libraries:
```bash
pip install -r requirements.txt
```

## Training

You can use the scripts in the bin directory to train a model.
```bash
bash bin/train_ours.sh
```

## Test 

We employ four metrics to measure the midline delineated by different methods, including line distance error (LDE), max shift distance error (MSDE), hausdorff distance (HD) and average symmetric surface distance (ASD).

```bash
bash bin/test_ours.sh
```

## Citation
If you use this code for your research, please cite our paper.

```
@inproceedings{wang2020context,
    title={Context-Aware Refinement Network Incorporating Structural Connectivity Prior for Brain Midline Delineation},
    author={Wang, Shen and Liang, Kongming and Li, Yiming and Yu, Yizhou and Wang, Yizhou},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
    pages={208--217},
    year={2020},
    organization={Springer}
    }
```
