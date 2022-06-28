# RCT-Random-Consistency-Training

Welcome to RCT-Random consistency training! This is the official implementation of RCT. RCT has already been accepcted by INTERSPEECH 2022.

[Paper :star_struck:](https://arxiv.org/abs/2110.11144) ![Paper :star_struck:](https://img.shields.io/badge/arXiv-2110.11144-brightgreen)**|** [Issues :sweat_smile:](https://github.com/Audio-WestlakeU/RCT-Random-Consistency-Training/issues)
 **|** [Lab :hear_no_evil:](https://github.com/Audio-WestlakeU) **|** [Contact :kissing_heart:](sao_year@126.com)

## Introduction

<div  align="center">    
<image src="/imgs/rct_structure.PNG"  width="500" alt="The structure of RCT" />

**RCT: Random Consistency Training**
</div>

RCT is a semi-supervised training secheme for Sound Event Detection (SED). But we believe it has a more generalized usage on other semi-supervised implementations!

RCT is constructed for SED, and we built it based on the baseline model for DCASE 2021 challenge. Please refer
to [[1]](https://http://arxiv.org/abs/2110.11144) and [[2]](https://github.com/DCASE-REPO/DESED_task) for more details about the code architecture. The model is built based on [PytorchLightning](https://www.pytorchlightning.ai/), if you are not familiar with its workflow, you can just focus: 1. the `training_step()` in `sed_trainer_rct.py`; 2. the `class RandAugment` in `rand_augm_agg.py`, to understand RCT.

## Training

The training/validation data is obtained from the DCSAE2021 task4 [DESED dataset](https://github.com/turpaultn/DESED).
The download of DESED is quite tedious and not all data is available for the accesses. You could ask for help from the DCASE committee to get the full dataset. Noted that, your testing result might be different with an incomplete validation dataset.

To train the model, please first get the baseline architecture of [DCASE2021 task 4](https://github.com/DCASE-REPO/DESED_task)
by:
```bash
git clone git@github.com:DCASE-REPO/DESED_task.git
```
Don't forget to configure your environment by their requirements.

After complete the above setup, you could add the codes of this repo to the baseline repo.

```bash
git clone git@github.com:Audio-WestlakeU/RCT-Random-Consistency-Training.git
```

To train the model, **DO NOT** forget to change your dataset path in `recipes/dcase2021_task4_baseline/confs/sed_rct.yaml`
to **YOUR_PATH_TO_DESED**. Then, please run:
```bash
python train_sed_rct.py
```

If you want to customize your training, you could modify the configuration file in
`recipes/dcase2021_task4_baseline/confs/sed_rct.yaml`. We provided our own implementations of different data augmentations including
[SpecAug](https://arxiv.org/pdf/1904.08779.pdf?source=post_page---------------------------),
[FilterAug](https://github.com/frednam93/FilterAugSED), pitch shift and time shift.

Using the proposed self-consistency loss is set as a trigger in `sed_rct.yaml` by
```angular2html
augs:    
    consis: True
```

Of course, we encourage the implementation of other data augmentations to be added and tested using RCT.

## Results
The result of a single model of RCT is around 40.12% and 61.39% for PSDS 1 and PSDS 2 under 7 trials.
You may get higher or lower results according to your choice of seeds. We provided the results of 3 trials:
<div align="center">

| Trial num. | Seed | PSDS_1 | PSDS_2 |
| :--------: | :--: | :----: | :----: |
|     1      |  42  | 39.69% | 61.59% |
|     2      |   1  | 40.49% | 62.67% |
|     3      |   2  | 39.50% | 60.03% |
</div>





## Reference
[1] DESED Dataset: https://github.com/turpaultn/DESED

[2] DCASE2021 Task4 baseline: https://github.com/DCASE-REPO/DESED_task

[3] SpecAug: https://arxiv.org/pdf/1904.08779

[4] FilterAug: https://github.com/frednam93/FilterAugSED
