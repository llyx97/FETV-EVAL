


# Text2Video-Zero

This repository is the implementation of [Text2Video-Zero](https://arxiv.org/abs/2303.13439), modified from the [official implementation](https://github.com/Picsart-AI-Research/Text2Video-Zero).


## Setups
Download the model checkpoints from [huggingface](https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0/tree/main) to the folder `sd_models/dreamlike-art`.

Install requirements using Python 3.9 and CUDA >= 11.6
```
virtualenv --system-site-packages -p python3.9 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Inference
```
bash run.sh
```

## License
This code is published under the CreativeML Open RAIL-M license. The license provided in this repository applies to all additions and contributions we make upon the original stable diffusion code. The original stable diffusion code is under the CreativeML Open RAIL-M license, which can found [here](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE).


## BibTeX
```
@article{text2video-zero,
    title={Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators},
    author={Khachatryan, Levon and Movsisyan, Andranik and Tadevosyan, Vahram and Henschel, Roberto and Wang, Zhangyang and Navasardyan, Shant and Shi, Humphrey},
    journal={arXiv preprint arXiv:2303.13439},
    year={2023}
}
```
