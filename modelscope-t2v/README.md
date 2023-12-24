# Implementation of ModelScopeT2V
This repo is the implementation of [ModelScopeT2V](https://arxiv.org/abs/2308.06571), adapted from the official implementation in [ModelScope](https://modelscope.cn/models/damo/text-to-video-synthesis/summary).

## Setups
Download all the files from [ModelScope](https://modelscope.cn/models/damo/text-to-video-synthesis/files) to the folder `weights`.
Install requirements:

```
  pip install modelscope==1.4.2
  pip install open_clip_torch
  pip install pytorch-lightning
```

## Run Inference
```
  bash run.sh
```
