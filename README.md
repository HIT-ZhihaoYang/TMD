# <p align=center> TMD: Text-Assisted Motion-Topology Decoupling Framework for  Skeleton-Based Temporal Action Segmentation</p>



## Introduction
> * This implementation code encompasses both training `train.py` and evaluation `evaluation.py` procedures.
> * A single GPU (NVIDIA RTX 3090) can perform all the experiments.

## Enviroment
Pytorch == `'2.5.1+cu124'`, 
torchvision == `0.20.1`, 
python == `3.10.15`, 
CUDA==`12.2`

### Enviroment Setup
Within the newly instantiated virtual environment, execute the following command to install all dependencies listed in the `requirements.txt` file.

``` python
pip install -r requirements.txt
```

## Datasets
All datasets can be downloaded.
**Note**：These datasets have been openly collected, curated, and subsequently released on a cloud platform by the authors of "A Decoupled Spatio-Temporal Framework for Skeleton-based Action Segmentation", rather than provided by the authors of the present paper.

Gratitude is extended to the respective authors for their contributions in providing and consolidating these datasets.



## Preparation

Orgnize the folder in the following structure (**Note**: please check it carefully):

> * The `result` folder and its contents will be automatically generated during code execution (`result` is the default storage path for results).


```
|-- config/
|   |-- MCFS-130/
|   |   -- config.yaml
|-- csv/
|-- dataset/
|   |-- MCFS-130/
|   |   |-- features/
|   |   |-- groundTruth/
|   |   |-- gt_arr/
|   |   |-- gt_boundary_arr/
|   |   |-- splits/
|   |   |-- mapping.txt
|--embeddings/
|-- libs/
|-- result/
|   |-- MCFS-130/
|   |   |-- split1/
|   |   |   |-- best_test_model.prm
|-- text/
|-- utils/
|-- train.py
|-- evaluate.py

```

## Get Started

### Training

To train our model on different datasets, use the following command:

```shell
python train.py 
```

Here, `--dataset` can be one of the following: LARA, MCFS-22, MCFS-130, PKU-subject, or PKU-view. 
`--cuda` specifies the ID number of the GPU to be used for training. 
Additionally, you can use `--result_path` to specify the output path, which defaults to `./result`.

If you wish to modify other parameters, please make the necessary changes in `csv/PKU-subject/config.yaml`.



### Evaluation

To evaluate the performance of the results obtained after running the training:

```shell
python evaluate.py
```

Here, `--dataset` and `--cuda` have the same meaning as in the training command. 
Note that if you specify the evaluation `--result_path`, it should match the training `--result_path`, which defaults to `./result`.

## Acknowledgement

Our work is closely related to the following assets that inspire our implementation. We gratefully thank the authors. 

- DeST:  https://github.com/lyhisme/DeST
- LaSA:  https://github.com/HaoyuJi/LaSA
- MoMA-M3T:  https:// github.com/ kuanchihhuang/ MoMA- M3T
- AMGCFN:  https:// github.com/ kuanchihhuang/ MoMA- M3T




