# 3D Novel View Synthesis (JAX)
This is an implementation of 3DiM ["Novel View Synthesis with Diffusion Models"](https://arxiv.org/pdf/2210.04628.pdf) using **JAX with distributed training on multiple GPUs**.

3DiM is a **diffusion model for 3D novel view synthesis**, which is able to translate a single input view into consistent and sharp completions across many views. The core component of 3DiM is a **pose-conditional image-to-image diffusion model**, which takes a source view and its pose as inputs, and generates a novel view for a target pose as output. This is a basic implementation of the method with *k=1* conditioning. 

More details about the work can be found [here](https://3d-diffusion.github.io/).

<p align="center">
<img src="https://user-images.githubusercontent.com/33219837/212821999-fbb947a1-a56c-48c7-8945-ddeacc6496c2.png" data-canonical-src="https://3d-diffusion.github.io/" width="300" height="300" />
</p>

Training is done using **JAX and FLAX**. The training can be **distributed** across multiple devices automatically if available. Since JAX does not have an inbuilt dataloader, we use `torch.dataset` for data operations. The dataloader has been adopted from [Scene Representation Networks](https://github.com/vsitzmann/scene-representation-networks). 

## Installation

```
git clone https://github.com/shiveshkhaitan/novel_view_synthesis_3d
cd instadeep_ml_test
```

### Nvidia docker
The package supports docker installation. To enable GPUs for docker, see installation guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installing-on-ubuntu-and-debian)

### Build docker
```
docker build -f Dockerfile . -t 3dim
```

## Usage

To start training
```
docker run -it --rm --memory '16g' --shm-size '16g' --gpus all \
--mount type=bind,source=$PWD,target=/home/3dim 3dim \
bash -c 'python3 train.py'
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
Currently this model is able to successfully denoise noisy inputs. However, it is not powerful enough to create novel views during sampling.
