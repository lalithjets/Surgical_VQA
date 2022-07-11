<div align="center">

<samp>

<h2> Surgical-VQA: Visual Question Answering in Surgical Scenes using Transformer </h1>

<h4> Lalithkumar Seenivasan*, Mobarakol Islam*, Adithya K Krishna and Hongliang Ren </h3>

</samp>   

---
| **[ [```arXiv```](<https://arxiv.org/abs/2206.11053>) ]** |**[ [```Paper```](<>) ]** |
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

<p align="center">
<img src="figures/SurgicalVQA.jpg" alt="SurgicalVQA" width="1000"/>
</p>


## VisualBert ResMLP for Classification and Sentence Generation
In our proposed VisualBERT ResMLP encoder model, we aim to further boost the interaction between the input tokens for vision-and-language tasks. The VisualBERT [1] model relies primarily on its self-attention module in the encoder layers to establish dependency relationships and allow interactions among tokens. Inspired by residual MLP (ResMLP) [2], the intermediate and output modules of the VisualBERT [15] model are replaced by cross-token and cross-channel modules to further enforce interaction among tokens. In the cross-token module, the inputs word and visual tokens are transposed and propagated forward, allowing information exchange between tokens. The resultant is then transposed back to allow per-token forward propagation in the cross-channel module. Both cross-token and cross-channel modules are followed by element-wise summation
with a skip-connection (residual-connection), which are layer-normalized. 

<p align="center">
<img src="figures/architecture.jpg" alt="architecture" width="1000"/>
</p>

## Results

<p align="center">
<img src="figures/Qualitative_analysis.jpg" alt="Qualitative_analysis" width="1000"/>
</p>

## Conclusion
We design a Surgical-VQA algorithm to answer questionnaires on surgical tools, their interactions and surgical procedures based on our two novel SurgicalVQA datasets evolved from two public datasets. To perform classification and sentence-based answering, vision-text attention-based transformer models are employed. A VisualBERT ResMLP transformer encoder model with lesser model parameters is also introduced that marginally outperforms the base vision-text attention encoder model by incorporating a cross-token sub-module. The influence of the number of input image patches and the inclusion of temporal visual features on the model’s performance is also reported. While our Surgical-VQA task answers to less-complex questions, from the application standpoint, it unfolds the possibility of incorporating open-ended questions where the model could be trained to answer surgery-specific complex questionnaires. From the model standpoint, future work could focus on introducing an asynchronous training regime to incorporate the benefits of the cross-patch sub-module without affecting the self-attention sub-module in sentence-based answer-generation tasks.

---

### Conda Environment

```bash
conda env create --name svqa --file=env.yml
```

---
## Directory Setup
<!---------------------------------------------------------------------------------------------------------------->
In this project, we implement our method using the Pytorch and DGL library, the structure is as follows: 

- `checkpoints/`: Contains trained weights.
- `dataloaders/`: 
    - dataloaderClassification.py : Dataloader for classification task.
    - dataloaderSentence.py : Dataloader for sentence task.
- `dataset/`
    - `bertvocab/`
        - `v2` : bert tokernizer
    - `Cholec80-VQA/`
        - `Classification`: 1-40 videos Classification Q&A pairs
            - `1`
            - `.....`
            - `40`
        - `Sentence`: 1-40 videos Sentence Q&A pairs
            - `1`
            - `.....`
            - `40`
        - `cropped_images`: contains 1-40 video folders.
            - `1`: contains image frames extracted from video.
                - `vqa`:  Contains img_features extracted from each frame with different patch size.
                    - `img_features`
                        - `1x1`: img_features extracted with a patch size of 1x1
                        - `2x2`: img_features extracted with a patch size of 2x2
                        - `3x3`: img_features extracted with a patch size of 3x3
                        - `4x4`: img_features extracted with a patch size of 4x4
                        - `5x5`: img_features extracted with a patch size of 5x5
                - `0.png`
                - `1.png`
                - `.....`
            - `.....`
            - `40`
    - `EndoVis-18-VQA/` : seq_{1-7,9-12,14-16}. Each sequence folder follows the following seq_1 folder structure. 
        - `seq_1`: 
            - `left_frames`: Image frames (left_frames) for each sequence can be downloaded from  EndoVIS18 challange.
            - `vqa`
                - `Classification`: Classification Q&A pairs.
                - `Sentence`: Sentence Q&A pairs.
                - `img_features`: Contains img_features extracted from each frame with different patch size.
                    - `1x1`: img_features extracted with a patch size of 1x1
                    - `2x2`: img_features extracted with a patch size of 2x2
                    - `3x3`: img_features extracted with a patch size of 3x3
                    - `4x4`: img_features extracted with a patch size of 4x4
                    - `5x5`: img_features extracted with a patch size of 5x5
        - `....`
        - `seq_16`

    - `VQA-Med/`
    - `feature_extraction_Cholec80-VQA.py`: Used to extract features (based on patch size) for Cholec80-VQA images.

- `models/`: 
    - VisualBertResMLP.py : Our proposed encoder.
    - visualBertClassification.py : VisualBert encoder-based classification model.
    - VisualBertResMLPClassification.py : VisualBert ResMLP encoder-based classification model.
    - VisualBertSentence.py : VisualBert encoder + Transformer decoder sentence generation model.
    - VisualBertResMLPSentence.py : VisualBert ResMLP encoder + Transformer decoder sentence generation model.
- train_classification.py
- test_sentence.py
- eval_sentence.py
- utils.py

---
## Dataset

1. Med-VQA (C1, C2 & C3)
    - Image frame and question & answer pairs - **[[`MedFuse Med-VQA Dataset (link to public dataset will be added shortly)`]()]**
2. EndoVis-18-VQA **[[`EndoVis-18-VQA Q&A pair annotation`](https://drive.google.com/drive/folders/1hu_yK27Xz2_lvjjZ97-WF2MK_JO14MWI?usp=sharing)]**
    - Images
    - Classification
    - Sentence
3. Cholec80-VQA **[[`Cholec80-VQA Q&A pair annotation`](https://drive.google.com/drive/folders/1QAqCEi_tMY9n3b37on0w6vr03cHom2ls?usp=sharing)]**
    - Images
    - Classification Task: Question & answer pairs annotation
    - Sentence Task: Question & answer pairs annotation
---

### Run training
- Classification
    - Arguments: 
        - TBA
    
    ```bash
    python3 train_classification.py
    ```

- Sentence
    - Arguments: 
        - TBA
    
    ```bash
    python3 train_sentence.py
    ```

---
## Evaluation
- Sentence    
    - Arguments: 
        - TBA
    
    ```bash
    python3 evaluation.py
    ```

---
## References
Code adopted and modified from :
1. Official implementation of VisualBertModel
    - Paper [VISUALBERT: A SIMPLE AND PERFORMANT BASELINE FOR VISION AND LANGUAGE](https://arxiv.org/pdf/1908.03557.pdf).
    - official pytorch implementation [Code](https://github.com/huggingface/transformers.git).

2. ResMLP
    - Paper [ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/pdf/2105.03404.pdf%E2%80%8Barxiv.org).
    - Official Pytorch implementation [Code](https://github.com/facebookresearch/deit.git).

---

## Contact
For any queries, please contact [Lalithkumar](mailto:lalithjets@gmail.com).

---
