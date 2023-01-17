# 3D Novel View Synthesis
Implementation of 3DiM ["NOVEL VIEW SYNTHESIS WITH DIFFUSION MODELS"](https://arxiv.org/pdf/2210.04628.pdf). 3DiM is a diffusion model for 3D novel view synthesis, which is able to translate a single input view into consistent and sharp completions across many views. The core component of 3DiM is a pose-conditional image-to-image diffusion model, which takes a source view and its pose as inputs, and generates a novel view for a target pose as output. This is a basic implementation of the method with k=1 conditioning. 

https://3d-diffusion.github.io/

![view_synthesis](https://user-images.githubusercontent.com/33219837/212821999-fbb947a1-a56c-48c7-8945-ddeacc6496c2.png "Source: https://3d-diffusion.github.io/")

Training is done using JAX and FLAX. Since JAX does not have an inbuilt dataloader, we use torch.dataset for data operations. The dataloader has been adopted from [Scene Representation Networks](https://github.com/vsitzmann/scene-representation-networks). 

To start training
```
python3 train.py
```

A smaller model with the following hyperparameters is available [here](https://drive.google.com/file/d/1SEVgheRjBq3AdLMpxhnYQP0unfS0LA55/view?usp=sharing). 
```
	ch: int = 32
	ch_mult = (1, 2,)
	emb_ch: int = 32
	num_res_blocks: int = 2
	attn_resolutions = (8, 16, 32)
	attn_heads: int = 4
	batch_size: 8
	image_sidelength: 64
```
This model is able to successfully denoise noisy inputs. However, it is not powerful enough to create novel views during sampling.
