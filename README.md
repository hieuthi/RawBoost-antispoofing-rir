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
 Traditional  33.65  23.33  22.36  22.51  22.51  18.58  17.54  18.27  18.66  21.14
 Wav.Concat.  31.23  27.51  27.50  29.39  29.29  18.09  18.18  18.27  18.27  22.60
  Neural AR   34.95  30.17  30.13  30.71  31.23  19.06  19.45  19.50  19.32  24.66
Neural non-AR 35.21  30.80  30.75  30.98  31.42  19.49  20.14  20.24  19.80  25.25
   Unknown    36.85  28.78  28.63  28.79  29.09  19.28  19.46  19.80  19.67  24.62
   Pooled     34.60  27.26  27.03  27.50  27.68  18.93  18.98  19.18  19.10  23.45


===============
Table for Threshold
===============

                 C1       C2       C3       C4       C5       C6       C7       C8       C9     Pooled
 Traditional  0.000117 0.000085 0.000081 0.000083 0.000082 0.000117 0.000118 0.000128 0.000118 0.000102
 Wav.Concat.  0.000109 0.000099 0.000097 0.000100 0.000099 0.000113 0.000125 0.000128 0.000116 0.000110
  Neural AR   0.000124 0.000105 0.000104 0.000104 0.000104 0.000124 0.000139 0.000146 0.000128 0.000121
Neural non-AR 0.000125 0.000108 0.000107 0.000105 0.000105 0.000130 0.000152 0.000154 0.000132 0.000126
   Unknown    0.000132 0.000102 0.000100 0.000099 0.000098 0.000127 0.000139 0.000148 0.000132 0.000121
   Pooled     0.000122 0.000098 0.000095 0.000095 0.000094 0.000122 0.000132 0.000140 0.000125 0.000114

```

### Probing RIR vulnarable
Using the DF set with C1R2.tgz reverberant set to probing RIR vulnarable
Go to the eval-package directory than run
```
python eval_C1R2.py ../eval_CM_scores_file.txt <threshold>
```

with `<threshold>` is the threshold value of C1-Pooled obtained from previous step. Examples
```
False Acceptance Rate (overall): 34.599%
1.75  67.96 63.99 57.89 38.32 24.33 23.73 20.02 19.64
1.5   82.26 63.33 55.71 39.33 27.68 23.45 17.70 18.38
1.25  74.04 60.26 56.18 39.28 30.95 25.46 20.69 20.62
1.0   59.69 61.78 54.55 39.80 30.05 22.28 22.49 20.26
0.75  73.28 62.61 53.31 37.84 25.56 22.88 22.23 18.40
0.5   66.39 65.89 53.16 39.98 26.82 25.38 21.46 20.80
0.25  78.95 68.01 56.90 40.98 27.11 22.16 21.43 19.77
0  76.85 67.00 51.17 39.55 28.20 23.21 22.20 20.70
FAR   -10 -5 0 5 10 15 20 25
```



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

