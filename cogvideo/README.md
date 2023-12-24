# CogVideo

This repository is modified from the [official implementation](https://github.com/THUDM/CogVideo) of [CogVideo](http://arxiv.org/abs/2205.15868).

```
@article{hong2022cogvideo,
  title={CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers},
  author={Hong, Wenyi and Ding, Ming and Zheng, Wendi and Liu, Xinghan and Tang, Jie},
  journal={arXiv preprint arXiv:2205.15868},
  year={2022}
}
```

## Getting Started

### Setup

* Hardware: Linux servers with Nvidia A100s are recommended, but it is also okay to run the pretrained models with smaller `--max-inference-batch-size` and `--batch-size` or training smaller models on less powerful GPUs.
* Environment: install dependencies via `pip install -r requirements.txt`. 
* LocalAttention: Make sure you have CUDA installed and compile the local attention kernel.

```shell
git clone https://github.com/Sleepychord/Image-Local-Attention
cd Image-Local-Attention && python setup.py install
```

### Download

Our code will automatically download or detect the models into the path defined by environment variable `SAT_HOME`. You can also manually download [CogVideo-Stage1](https://lfs.aminer.cn/misc/cogvideo/cogvideo-stage1.zip) and [CogVideo-Stage2](https://lfs.aminer.cn/misc/cogvideo/cogvideo-stage2.zip) and place them under SAT_HOME (with folders named `cogvideo-stage1` and `cogvideo-stage2`)

### Text-to-Video Generation

```
bash run.sh
```

Arguments useful in inference are mainly:

* `--input-source [path or "interactive"]`. The path of the input file with one query per line. A CLI would be launched when using "interactive".
* `--output-path [path]`. The folder containing the results.
* `--batch-size [int]`. The number of samples will be generated per query.
* `--max-inference-batch-size [int]`. Maximum batch size per forward. Reduce it if OOM. 
* `--stage1-max-inference-batch-size [int]` Maximum batch size per forward in Stage 1. Reduce it if OOM. 
* `--both-stages`. Run both stage1 and stage2 sequentially. 
* `--use-guidance-stage1` Use classifier-free guidance in stage1, which is strongly suggested to get better results. 

You'd better specify an environment variable `SAT_HOME` to specify the path to store the downloaded model.

*Currently only Chinese input is supported.*
