<div align="center">

<samp>

<h2> Surgical-VQA: Visual Question Answering in Surgical Scenes using Transformer </h1>

<h4> Lalithkumar Seenivasan*, Mobarakol Islam*, Adithya Krishna and Hongliang Ren </h3>

</samp>   

---
| **[ [```arXiv```](<>) ]** |**[ [```Paper```](<>) ]** |
|:-------------------:|:-------------------:|
    
The International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2022
---

</div>     
    
---

If you find our code or paper useful, please cite as

```bibtex

```

---
## Abstract
Visual question answering (VQA) in surgery is largely unexplored. Expert surgeons are scarce and are often overloaded with clinical and academic workloads. This overload often limits their time answering questionnaires from patients, medical students or junior residents related to surgical procedures. At times, students and junior residents also refrain from asking too many questions during classes to reduce disruption. While computer-aided simulators and recording of past surgical procedures have been made available for them to observe and improve their skills, they still hugely rely on medical experts to answer their questions. Having a Surgical-VQA system as a reliable ‘second opinion’ could act as a backup and ease the load on the medical experts in answering these questions. The lack of annotated medical data and the presence of domain-specific terms has limited the exploration of VQA for surgical procedures. In this work, we design a Surgical-VQA task that answers questionnaires on surgical procedures based on the surgical scene. Extending the  MICCAI endoscopic vision challenge 2018 dataset and workflow recognition dataset further, we introduce two Surgical-VQA datasets with classification and sentence-based answers. To perform Surgical-VQA, we employ vision-text transformers models. We further introduce a residual MLP-based VisualBert encoder model that enforces interaction between visual and text tokens, improving performance in classification-based answering. Furthermore, we study the influence of the number of input image patches and temporal visual features on the model performance in both classification and sentence-based answering.

## Introduction
TBA 

## Dataset
TBA

## Model
TBA 

## Results
TBA 

## Conclusion
TBA 

---

## Library Prerequisities
TBA

## Setup (From an Env File)
TBA
<!-- We have provided environment files for installation using conda -->

### Using Conda
TBA
<!-- ```bash
conda env create -f environment.yml
``` -->

---
## Directory setup
<!---------------------------------------------------------------------------------------------------------------->
In this project, we implement our method using the Pytorch and DGL library, the structure is as follows: 

- `checkpoints/`: Contains trained weights.
- `dataloaders/`: Contains dataloader for classification and sentence model.
- `dataset/`
    - `bertvocab/` : TBA
    - `cholec80/` : TBA
    - `faimed3d/` : TBA
    - `instrument18/` : TBA
    - `VQA-Med/` : TBA
- `models/`: Contains network models.

---
## Dataset

### Classification
1. Med-VQA (C1, C2 & C3)
    - Images
    - Question and answer pairs : Download **[[`annotations`]()]**
2. EndoVis-18-VQA (C)
    - Images : Download **[[`Challenge Portal`](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/home/)]** 
    - Question and answer pairs : Download **[[`annotations`]()]**
3. Cholec80-VQA (C)
    - Images
    - Question and answer pairs : Download **[[`annotations`]()]**

### Sentence
1. EndoVis-18-VQA (S)
    - Images : Download **[[`Challenge Portal`](https://endovissub2018-roboticscenesegmentation.grand-challenge.org/home/)]** 
    - Question and answer pairs : Download **[[`annotations`]()]**
2. Cholec80-VQA (S) 
    - Images
    - Question and answer pairs : Download **[[`annotations`]()]**

---

### Run training

- Classification

```bash
python3 train_classification.py
```

- Sentence
```bash
python3 train_sentence.py
```

---
## Evaluation


```bash
python3 evaluation.py
```

---
## Acknowledgement
Code adopted and modified from :
1. Official implementation of VisualBertModel
    - Paper [TBA]().
    - official pytorch implementation [huggingface/transformers](https://github.com/huggingface/transformers.git).

2. ResMLP
    - Paper [TBC]().
    - Official Pytorch implementation [code]().

---

## Contact
For any queries, please contact [Lalithkumar](mailto:lalithjets@gmail.com).

---
