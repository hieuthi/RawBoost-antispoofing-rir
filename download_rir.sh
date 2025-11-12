#!/bin/bash

if ! which wget >/dev/null; then
  echo "$0: wget is not installed."
  exit 1;
fi


mkdir downloads
wget https://zenodo.org/api/records/17180301/files/C1R1.tgz/content -O downloads/C1R1.tgz
wget https://zenodo.org/api/records/17180301/files/C1R2.tgz/content -O downloads/C1R2.tgz
wget https://zenodo.org/api/records/17180301/files/C1R1_info.csv/content -O downloads/C1R1_info.csv
wget https://zenodo.org/api/records/17180301/files/C1R2_info.csv/content -O downloads/C1R2_info.csv
wget https://zenodo.org/api/records/17180301/files/rir-syn-test.tgz/content -O downloads/rir-syn-test.tgz
wget https://zenodo.org/api/records/17180301/files/rir-syn-train.tgz/content -O downloads/rir-syn-train.tgz
wget https://zenodo.org/api/records/17180301/files/rir-syn-test_info.csv/content -O downloads/rir-syn-test_info.csv
wget https://zenodo.org/api/records/17180301/files/rir-syn-train_info.csv/content -O downloads/rir-syn-train_info.csv


mkdir RIRs
tar -xvf downloads/rir-syn-train.tgz -C downloads/
mv downloads/rir-syn-train/ RIRs

find RIRs/rir-syn-train/wav16k/ -name "*.wav" > RIRs/rir-syn-train.scp

