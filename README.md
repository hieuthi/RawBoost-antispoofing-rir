# Modified

### Training with RIRs
To train the model with augmented RIRs run:
```
# Download RIRs
bash download_rir.sh

# Train with RIRs
python main_reverb.py --track=LA --loss=WCE   --lr=0.0001 --batch_size=128 --rirscp RIRs/rir-syn-train.scp --probability 0.99 --name_tag '_reverb_syn_0.99'
```

### Testing with reverberant speech
To test with reverberant fake speech simply put the flac files in C1R1.tgz or C1R2.tgz to `ASVspoof2021_DF_eval/flac`.

Then you can testing DF set as usual

```
python main_reverb.py --track=DF --loss=WCE --is_eval --eval --model_path='/path/to/your/best_model.pth' --eval_output='eval_CM_scores_file.txt'
```

### Evaluating
From eval-package directory run below script to evaluating DF set.

First you need to download DF keys then uncompressed to eval-packages/keys

```
# Download keys for DF
bash download.sh

# Run evaluation on DF
python eval_DF.py ../eval_CM_scores_file.txt
```

The results contain two tables EER and Thresholds. Examples
```
===============
Table for EERs
===============

                C1     C2     C3     C4     C5     C6     C7     C8     C9   Pooled
 Traditional  26.23  24.14  23.44  23.04  23.03  17.36  16.92  17.31  17.44  20.38
 Wav.Concat.  26.79  28.36  29.76  28.88  29.24  17.54  17.72  17.72  17.83  22.27
  Neural AR   31.68  33.41  33.22  33.09  33.66  19.97  19.88  20.10  19.80  25.50
Neural non-AR 29.50  31.31  31.19  30.72  31.32  19.49  19.45  19.71  19.58  24.37
   Unknown    30.28  29.29  29.45  28.36  29.00  19.24  18.93  19.15  19.37  23.54
   Pooled     29.24  29.12  29.07  28.36  28.84  18.93  18.53  18.84  19.06  23.14


===============
Table for Threshold
===============

                 C1       C2       C3       C4       C5       C6       C7       C8       C9     Pooled
 Traditional  0.000024 0.000021 0.000021 0.000022 0.000022 0.000033 0.000036 0.000035 0.000034 0.000026
 Wav.Concat.  0.000025 0.000024 0.000026 0.000028 0.000027 0.000034 0.000041 0.000038 0.000037 0.000031
  Neural AR   0.000032 0.000031 0.000032 0.000035 0.000035 0.000053 0.000063 0.000059 0.000058 0.000043
Neural non-AR 0.000028 0.000028 0.000029 0.000031 0.000030 0.000048 0.000057 0.000054 0.000053 0.000038
   Unknown    0.000029 0.000025 0.000026 0.000028 0.000027 0.000044 0.000052 0.000047 0.000048 0.000035
   Pooled     0.000027 0.000025 0.000026 0.000028 0.000027 0.000041 0.000047 0.000045 0.000044 0.000033
```

### Probing RIR vulnarable
Using the DF set with C1R2.tgz reverberant set to probing RIR vulnarable
Go to the eval-package directory than run
```
python eval_C1R1.py ../eval_CM_scores_file.txt <threshold>
```

with <threshold> is the threshold value of C1-Pooled obtained from previous step



# RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing
===============
This repository contains our implementation of the paper, "RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing". This work introduce RawBoost, a data boosting and augmentation method for the design of more reliable spoofing detection solutions which operate directly upon raw waveform inputs ([Paper link here](https://arxiv.org/pdf/2111.04433.pdf)).


## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone https://github.com/TakHemlata/RawBoost-antispoofing.git
$ conda create --name RawBoost_antispoofing python=3.8.8
$ conda activate RawBoost_antispoofing
$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
$ pip install -r requirements.txt
```


## Experiments

### Dataset
Our experiments are performed on the logical access (LA) partition of the ASVspoof 2021 dataset (train on 2019 LA training and evaluate on 2021 LA evaluation database).

### Training
To train the model run:
```
python main.py --track=LA --loss=WCE   --lr=0.0001 --batch_size=128
```

### Testing

To evaluate your own model on LA evaluation dataset:

```
python main.py --track=LA --loss=WCE --is_eval --eval --model_path='/path/to/your/best_model.pth' --eval_output='eval_CM_scores_file.txt'
```

We also provide a pre-trained models. To use it you can run: 
```
python main.py --track=LA --loss=WCE --is_eval --eval --model_path='Pre_trained_models.pth' --eval_output='RawBoost_eval_CM_scores.txt'
```

This repository is built on our End-to-end RawNet2 CM system (ASVspoof2021 Challenge baseline).
- [ASVspoof 2021 Challenge baseline repo](https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2)


## Contact
For any query regarding this repository, please contact:
- Hemlata Tak: tak[at]eurecom[dot]fr
- Massimiliano Todisco: todisco[at]eurecom[dot]fr

## Citation
If you use RawBoost code in your research please use the following citation:

```bibtex
@inproceedings{tak2021rawboost,
  title={RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing},
  author={Tak, Hemlata and Kamble, Madhu and Patino, Jose and Todisco, Massimiliano and Evans, Nicholas},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2022}
}
```

