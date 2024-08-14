<h2 align="center">
<a href="https://ojs.aaai.org/index.php/AAAI/article/view/27905" target="_blank">PHFormer: Multi-Fragment Assembly Using Proxy-Level Hybrid Transformer</a>
</h2>

<h3 align="center">
AAAI 2024
</h3>

<p align="center">
<a href="https://scholar.google.com/citations?user=UClf9_0AAAAJ&hl=en" target="_blank">Wenting Cui</a>, 
<a href="https://scholar.google.com/citations?hl=en&user=udCF7q4AAAAJ" target="_blank">Runzhao Yao</a>, 
<a href="https://scholar.google.com/citations?user=r2bk4sQAAAAJ&hl=en" target="_blank">Shaoyi Du</a><sup>*</sup>,
</p>

<p align="center">
National Key Laboratory of Human-Machine Hybrid Augmented Intelligence, <br>
National Engineering Research Center for Visual Information and Applications, <br>
Institute of Artificial Intelligence and Robotics, <br>
Xi’an Jiaotong University, China, <br>
<sup>*</sup>Corresponding author. 
</p>

## Introduction

Fragment assembly involves restoring broken objects to their  original geometries, and has many applications, such as archaeological  restoration.  Existing learning based frameworks  have shown potential for solving part assembly problems  with semantic decomposition, but cannot handle such geometrical  decomposition problems.  In this work, we propose  a novel assembly framework, proxy level hybrid Transformer,  with the core idea of using a hybrid graph to model  and reason complex structural relationships between patches  of fragments, dubbed as proxies.  To this end, we propose  a hybrid attention module, composed of intra and inter  attention layers, enabling capturing of crucial contextual  information within fragments and relative structural knowledge  across fragments.  Furthermore, we propose an adjacency  aware hierarchical pose estimator, exploiting a decompose  and integrate strategy.  It progressively predicts adjacent  probability and relative poses between fragments, and then  implicitly infers their absolute poses by dynamic information  integration.  Extensive experimental results demonstrate  that our method eectively reduces assembly errors while  maintaining fast inference speed.


<div align="center">
    <img src="assets/img.png" alt="framework" width="700" >
</div>


## 🔧  Installation and Dataset

We mainly follow [multi part assembly](https://github.com/Wuziyi616/multi_part_assembly/blob/master/docs/install.md) to install required packages.

After you have set up the environment, you also should install custom CUDA ops for `knnsearch`:
```bash
cd multi_part_assembly/utils/pointops/
python setup.py install
```

We train and evaluate our model on the [breakind bad dataset](https://github.com/Breaking-Bad-Dataset/Breaking-Bad-Dataset.github.io/blob/main/README.md), please follow the instructions to download it. 

After downloading and processing all the data, please change the `_C.data_dir` in `configs/_base_/datasets/breaking_bad/XXX.py` to your data path.

Moreover, when you first run the code, Please use the `_C.pre_compute = True` and it will automatically compute the adjacent labels and save them. After you have saved all the labels, please modify the `_C.pre_compute = True` to `False`.

Code has been tested with Ubuntu 18.04, GCC 7.5.0, Python 3.8, PyTorch 1.10.2, CUDA 11.1 and cuDNN 8.0.5.


## 🚅 Training
To train a model, please run:
```bash
python scripts/train.py --cfg_file configs/phformer/phformer-32x1-cosine_400e-everyday.py --fp16 --cudnn
```


## ⛳ Testing
To evaluate the performance of the pre-trained models on `everyday`, use the following commands:
```bash
# everyday
python scripts/test.py --weight checkpoint/pretrain/everyday.ckpt --cfg_file configs/phformer/phformer-32x1-cosine_400e-everyday.py
```
To evaluate the performance of models on `artifact`, use the following commands:
```bash
# model trained from scratch
python scripts/test.py --weight checkpoint/pretrain/artifact_scratch.ckpt --cfg_file configs/phformer/phformer-32x1-cosine_400e-artifact.py --category all
```

```bash
# model pretrained on everyday
python scripts/test.py --weight checkpoint/pretrain/artifact_pretrain.ckpt --cfg_file configs/phformer/phformer-32x1-cosine_400e-artifact.py --category all
```

To evaluate the generalization ability of models on `other`, use the following commands:
```bash
# model trained on everyday
python scripts/test.py --weight checkpoint/pretrain/everyday.ckpt --cfg_file configs/phformer/phformer-32x1-cosine_400e-other.py --category all
```

```bash
# model pretrained on everyday and then trained on artifact
python scripts/test.py --weight checkpoint/pretrain/artifact_pretrain.ckpt --cfg_file configs/phformer/phformer-32x1-cosine_400e-other.py --category all
```
Experimental results will be saved in a log file in the model folder.

## Visualization
If you want to visualize the assembly results, please use the following command:
```bash
python scripts/vis.py --weight checkpoint/pretrain/everyday.ckpt --cfg_file configs/phformer/phformer-32x1-cosine_400e-everyday.py --vis 10
```
The predicted results will be saved in `vis` folder.

## Citation

```bibtex
@inproceedings{cui2024phformer,
  title={PHFormer: Multi-Fragment Assembly Using Proxy-Level Hybrid Transformer},
  author={Cui, Wenting and Yao, Runzhao and Du, Shaoyi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={2},
  pages={1408--1416},
  year={2024}
}
```

## Acknowledgements
We thank the authors of the following excellent works:
- [MultiPartAssembly](https://github.com/Wuziyi616/multi_part_assembly)
- [GeoTransformer](https://github.com/qinzheng93/GeoTransformer)



