# Base ubuntu docker image with nvidia
FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04

# Install pip
RUN apt-get update
RUN apt-get -y install python3-pip
RUN apt-get -y --no-install-recommends install libgl1-mesa-glx libglib2.0-0

# Copy the requirements.txt file
COPY requirements.txt .

# Install requirements
RUN pip3 install -r requirements.txt
RUN pip3 install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Directory to mount package
RUN mkdir /home/3dim
