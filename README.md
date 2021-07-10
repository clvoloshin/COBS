# Caltech OPE Benchmarking Suite (COBS)

## Introduction

COBS is an Off-Policy Policy Evaluation (OPE) Benchmarking Suite. The goal is to provide fine experimental control to carefully tease out an OPE method's performance across many key conditions. 

We'd like to make this repo as useful as possible for the community. We commit to continual refactoring and code-review to make sure the COBS continues to serve its purpose. Help is always appreciated!

COBS is based on Empirical Study of Off Policy Policy Estimation paper (https://arxiv.org/abs/1911.06854)

## Getting started

### Tutorial

To get started using the experimental tools see [Tutorial.ipynb](https://github.com/clvoloshin/COBS/blob/master/Tutorial.ipynb)

### Installation

Tested on python3.6, python3.7.
```
python3.7 -m venv cobs-env
source cobs-env/bin/activate
pip3 install -r requirements.txt
```

## <a name="CitingCOBS"></a>Citing COBS

If you use COBS, please use the following BibTeX entry.

```
  @misc{voloshin2019empirical,
    title={Empirical Study of Off-Policy Policy Evaluation for Reinforcement Learning},
    author={Cameron Voloshin and Hoang M. Le and Nan Jiang and Yisong Yue},
    year={2019},
    eprint={1911.06854},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```


