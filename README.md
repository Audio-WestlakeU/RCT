# RCT-Random-Consistency-Training

Welcome to RCT-Random consistency training! This is the official implementation of RCT.

[Paper](google.com) :star_struck: **|** [Issues](https://github.com/Audio-WestlakeU/RCT-Random-Consistency-Training/issues) :sweat_smile:
 **|** [Lab](https://github.com/Audio-WestlakeU) :hear_no_evil: **|** [Contact](sao_year@126.com) :kissing_heart:

## Introduction

<div  align="center">    
<image src="/imgs/rct_structure.PNG"  width="500" alt="The structure of RCT" />

RCT: Random Consistency Training
</div>

RCT is a semi-supervised training secheme for Sound Event Detection (SED). But we believe it has a more generalized
usage for other applications!

Since the code is constructed for SED, we build RCT based on the baseline model for DCASE 2021 challenge. Please refer
to [[1]](https://github.com/turpaultn/DESED) and [[2]](https://github.com/DCASE-REPO/DESED_task) for more details. 

## Training

The training/validation data is obtained from the DCSAE2021 task4 [DESED dataset](https://github.com/turpaultn/DESED).
The downloading of the dataset is quite complicated, no all data is available for the accesses. So, your testing result might
be different with an incomplete validation dataset.

To train the model, please first get the baseline architecture of [DCASE2021 task 4](https://github.com/DCASE-REPO/DESED_task)
by:
```bash
git clone git@github.com:DCASE-REPO/DESED_task.git
```
Don't forget to configure your environment as their requirements.

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
The result of a single model of RCT is aruond 40.12% and 61.39% for PSDS 1 and PSDS 2.
You may get higher or lower results according to your choice of seeds.

## Reference
[1] DESED Dataset: https://github.com/turpaultn/DESED

[2] DCASE2021 Task4 baseline: https://github.com/DCASE-REPO/DESED_task

[3] SpecAug: https://arxiv.org/pdf/1904.08779

[4] FilterAug: https://github.com/frednam93/FilterAugSED