# DistilBERT Malware Detection using the CICIDS2017 Dataset

## Table of Contents

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Results](#results)

## Introduction

This project utilizes the DistilBERT encoder architecture from huggingface.co that has been fine tuned with CICIDS2017 dataset to detect the occurrence of malware in network trafffic. 

## Technologies Used

- Transformers (huggingface.co)
- PyTorch
- Numpy
- Seaborn
- Matplotlib

## Results

The purpose of this project was to gain experience in utilizing the BERT architecture via HuggingFace. The results were promising with most classification metrics at 0.998 accuracy with the exception of a two malware types.  This was due to an imbalanced dataset and close to zero feature correlation.  

The training took two days on a single GPU (3090 with 24Gb Memory) and inference took about an hour on 20% of the dataset.