# Caltech OPE Benchmarking Suite (COBS)
COBS based on Empirical Study of Off Policy Policy Estimation paper (https://arxiv.org/abs/1911.06854)

## Installation

Please use Python 3.6+. 
```
pip install -r requirements.txt
pip install -e .
```

## Usage

To replicate results in the paper, please see instructions in paper.py

To run your own experiments, see example.py (or example2.py). Alternatively, you can modify paper.py to your use-case.

## Goal

I'd like to make this repo as useful as possible for the community which means there's still a lot of work to be done in getting it to a modular, more package-like place. Help is kindly appreciated.


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


