This is a fork of the official Ray project. The following instructions correspond to the template to fine-tune Llama2 series at `doc/source/templates/04_finetuning_llms_with_deepspeed` in this repository. Please look at `README.rst` for more details regarding the Ray project.

# [Domino] Set up instructions 

## Environment Requirements

Make sure that you have the right workspace and cluster environments. The workspace environment details are provided with the Domino AI Hub project template. Here is what we used for the cluster environment - 

Base image `anyscale/ray:2.7.0-py39-cu118`

Dockerfile instructions

```
USER root

RUN \
  groupadd -g 12574 ubuntu && \
  useradd -u 12574 -g 12574 -m -N -s /bin/bash ubuntu

RUN pip install accelerate \
                aim \
                datasets \
                deepspeed \
                evaluate \
                ipywidgets \
                matplotlib \
                mlflow \
                numpy \
                pandas \
                peft \
                scipy \
                tblib \
                transformers==4.31.0 \
                sentencepiece \ 
                filelock==3.12.2 \ 
                tqdm==4.64.1

RUN pip install -U --force-reinstall torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
RUN git clone https://github.com/timdettmers/bitsandbytes.git
WORKDIR bitsandbytes

RUN git checkout 0.42.0

# CUDA_VERSIONS in {110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 120}
# make argument in {cuda110, cuda11x, cuda12x}
# if you do not know what CUDA you have, try looking at the output of: python -m bitsandbytes
RUN CUDA_VERSION=118 make cuda11x
RUN python setup.py install --force

RUN sudo chmod 777 -R /home/ray
```

## Hardware Requirements 

You do not need A100s to make fine-tuning with large models work! We recommend setting up a HWT using `g5.4xlarge` instances. 

**Do not use the autoscaler feature while launching a workspace**. Instead, start your workspaces with a Ray cluster with the required number of nodes. For testing purposes, we  recommend the following minimal requirements (to ensure a judicious and cost optimal way to use utilize GPU):
| Llama2 Model| No of Workers | instance type | GPU per node | GPU Memory | CPU Memory |
|-------------|-----|---------------|--------------|------------|------------|
| 7B  | 6   | g5.4xlarge    | 1 x A10G     | 24 GB      | 64 GB      |
| 13B | 16  | g5.4xlarge    | 1 x A10G     | 24 GB      | 64 GB      |
| 70B | 32  | g5.4xlarge    | 1 x A10G     | 24 GB      | 64 GB      |

The defaults are configured in the script `run_llama_ft.sh`  in the section below (BS==BatchSize and ND=No of Workers)
```
# Batch size and node count
case $SIZE in
"7b")
    BS=16
    ND=6 # change the number of devices/workers if you need more resources 
    ;;
"13b")
    BS=16
    ND=16 # change the number of devices/workers if you need more resources 
    ;;
"70b")
    BS=8
    ND=32 # change the number of devices/workers if you need more resources
    ;;
*)
    echo "Invalid size: ${SIZE}"
    exit 1
    ;;
esac
```

Change the values accordingly to increase/decrease the Batch Size or No of Workers/Devices. Also, for this test we used the same Hardware Tier for all pod types as shown below. You can change this upon your discretion depending on your resource availability.

| Type of Pod| HW Tier Instance Type | CPU | Memory | GPU | GPU Memory (Based on GPU Chosen)|
|-------------|-----|---------------|--------------|------------|------------|
| Workspace | g5.4xlarge  | 6    | 55     | 1 x A10G     | 24 GB      |
| Ray Head | g5.4xlarge  | 6    | 55     | 1 x A10G     | 24 GB      |
| Ray Worker | g5.4xlarge  | 6    | 55     | 1 x A10G     | 24 GB      |

We also allocated enough storage space per worker (for example, 100 GiB). Note that you might also have to increase the Workspace volume size under Project Settings (for example, 200 GiB). 

> Note: Domino defines the maximum number of workloads a user can start using the central config parameter
> `com.cerebro.domino.computegrid.userExecutionsQuota.maximumExecutionsPerUser`. By default, this is set to 25. This parameter applies to all interactive
> workloads started by the user which means, workspaces and any cluster pods attached to the workspaces (head and worker nodes included). 
> For a single workspace, it means you can start a ray cluster with 23 workers assuming you have no other workspaces running. If you
> have 5 standalone workspaces running, and start a workspace with an attached Ray cluster, that cluster can have a maximum of 18 workers.
> Ask your admin to increase this limit if you want to start more workloads (for example, in case you are working with Llama2 70b). 

The following steps are from the official Ray project for your reference. You can also refer to the official Ray repository for more context. We have tested fine-tuning and made changes to the code so that it runs seamlessly on Domino. We also provide an inference script `Inference.ipynb` to try out your model from within a Domino workspace. 

# [Ray Project] Finetuning Llama-2 series models with Deepspeed, Accelerate, and Ray Train TorchTrainer
| Template Specification | Description |
| ---------------------- | ----------- |
| Summary | This template demonstrates how to perform full parameter fine-tuning for Llama-2 series models (7B, 13B, and 70B) using TorchTrainer with the DeepSpeed ZeRO-3 strategy. |
| Time to Run | ~14 min. for 7B for 1 epoch on 3.5M tokens. ~26 min for 13B for 1 epoch.  ~151 min for 70b for 1 epoch.|
| Minimum Compute Requirements | 6xg5.4xlarge for worker nodes for 7B model, 16xg5.4xlarge nodes for 13B model, and 32xg5.4xlarge (or 2xp4de.24xlarge) nodes for 70B|
| Cluster Environment | This template uses a docker image built on top of the latest Anyscale-provided Ray image using Python 3.9: [`anyscale/ray:latest-py39-cu118`](https://docs.anyscale.com/reference/base-images/overview). |

## Getting Started

For 7B, set up a Ray Cluster with the following settings:

|  Node Type | Num | 
|------------|-----|
| Head node  | 1   |
| Worker node| 6  | 

And launch the following script:

```
./run_llama_ft.sh --size=7b [--as-test]
```

The flag `--as-test` is for demo / testing purposes as it runs through only one forward and backward pass of the model. The model loading, and remote checkpointing would still run. 

Similarly for 13B you need a different compute config. 

|  Node Type | Num | 
|------------|-----|
| Head node  | 1   |
| Worker node| 16  | 

```
./run_llama_ft.sh --size=13b [--as-test]
```

Similarly, for 70B you need a different compute config. 

|  Node Type | Num | 
|------------|-----|
| Head node  | 1   |
| Worker node| 32  | 


```
./run_llama_ft.sh --size=70b [--as-test]
```

## What is happening under the hood?

### Downloading the pre-trained checkpoint on to all GPU nodes. 

The pre-trained models for these models is quite large (12.8G for 7B model and 128G for 70B model). In order to make loading these models faster, we have mirrored the weights on to an AWS S3 bucket which can result in up 10GB/s download speed if the aws configs are setup correctly. 

### Cloud storage

Similarly the checkpoints during training can be quite large and we would like to be able to save those checkpoints to the familiar huggingface format so that we can serve it conveniently. The fine-tuning script in this template uses Ray Train Checkpointing to sync the checkpoints created by each node back to a centralized cloud storage on AWS S3. The final file structure for each checkpoint will have a look similar to the following structure:

```
aws s3 ls s3://<bucket_path>/checkpoint_00000

├── .is_checkpoint
├── .metadata.pkl
├── .tune_metadata
├── _metadata.meta.pkl
├── _preprocessor
├── _preprocessor.meta.pkl
├── added_tokens.json
├── config.json
├── generation_config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer.model
└── tokenizer_config.json
```

After training we can use [Aviary](https://github.com/ray-project/aviary) to deploy our fine-tuned LLM by providing the checkpoint path stored on cloud directly.

### Creating the dataset

The main fine-tuning script is written in a general format that would require you to provide a `jsonl` file for train and test datasets in addition to a `json` file listing the special tokens used in your dataset. 

For example each row in your dataset might be formated like the following:

```
{"input": "<ASSISTANT>How can I help you?</ASSISTANT><USER>how is the weather?</USER>}
```

And the special tokens can be:

```
{"tokens": ["<ASSISTANT>", "</ASSISTANT>", "<USER>", "</USER>"]}
```

Depending on the dataset you want to finetune on, the tokenization and dataset pre-processing will likely need to be adjusted. The current code is configured to train on the Grade School Math 8k (GSM8K) dataset. By running the code below we create three files that are needed to launch the training script with. 

```
python create_dataset.py

>>> data/train.jsonl # 7.4k training data
>>> data/test.jsonl # 1.3k test data
>>> tokens.json # a list of special tokens
```

This dataset is trained with a context length of 512 which includes excessive padding to keep all samples limited to 512 tokens. This means that the training dataset has 3.5 M tokens.

### Launching fine-tuning

The script is written using Ray Train + Deepspeed integration via accelerate API. The script is general enough that it can be used to fine-tune all released sizes of Llama-2 models. 

The CLI for seeing all the options is:

```
python finetune_hf_llm.py --help
```

This script was tested across three model sizes on the following cluster configurations on Anyscale platform. 


| Model Size | Base HF Model ID             | Batch size per device | GPUs           | Time per epoch (min.) |
|------------|------------------------------|-----------------------|----------------|-----------------------|
| 7B         | `meta-llama/Llama-2-7b-hf`   | 16                    | 16x A10G (24G) | ~14 min.              |
| 13B        | `meta-llama/Llama-2-13b-hf`  | 16                    | 16x A10G (24G) | ~26 min.              |
| 70B        | `meta-llama/Llama-2-70b-hf`  | 8                     | 32x A10G (24G) | ~190 min.             |



### Guideline on how to pick node instances when A100s are not available.

Here is the suggested cluster config for each workload:

7B:

```
head_node_type:
  name: head_node_type
  instance_type: m5.xlarge

worker_node_types:
- name: gpu_worker
  instance_type: g5.4xlarge
  min_workers: 0
  max_workers: 16
  use_spot: false
```


13B:

```
head_node_type:
  name: head_node_type
  instance_type: m5.xlarge

worker_node_types:
- name: gpu_worker
  instance_type: g5.12xlarge
  min_workers: 0
  max_workers: 4
  use_spot: false
```

70B:

```
head_node_type:
  name: head_node_type
  instance_type: m5.xlarge

worker_node_types:
- name: gpu_worker
  instance_type: g5.48xlarge
  min_workers: 0
  max_workers: 4
  use_spot: false
```

There are two things that you should consider when choosint the cluster configurations:

1. CPU RAM requirement for optimizer state and parameter offloading

Deepspeed offers [Zero-offload](https://www.deepspeed.ai/tutorials/zero-offload/) which allows offloading the optimizer or parameter states to the CPU memory for more memory efficient training. We have enabled this by default in our deepspeed configs used for this workspace template. 

This method creates extra CPU RAM requirements on the machines. A rule of thumb for this implementation is that it needs O(18M/N*K) CPU RAM where M is the model size, N is the number shards, and K is the number of GPUs on a single machine. 

For example, for 70B model, on an 8xA100 machine with 8-way sharding you would need `18 * (70 / 8) * 8 = 1.26 TB` of CPU RAM. This is not available on a single machine of 8xA100s. But if we use two 8xA100 machines instead, with 16-way sharding we would need `18 * (70 / 16) * 8 = 630 GB` of CPU RAM which is accessible.

Another example: For 70B model, on an 4xA10G machine with 32-way sharding you would need `18 * (70 / 32) * 4 = 158 GB` of CPU RAM on each machine. If you use 8xA10G machines instead you would need `18 * (70 / 32) * 8 = 316 GB` of CPU RAM on each machine. 

So availability of enough CPU RAM is very important when using optimizer state offloading.

2. CPU RAM requirement during checkpointing

During checkpointing in the middle of training, we have to aggregate the weights from all the shards back to rank 0 so that it can save the model. We can also save the weights of each shard indepedently and aggregate the weights later offline. The extra CPU memory requirement would not get solved tho. 

Emprically the implementation that `accelerate` provides needs `O(4M)` CPU RAM on rank 0 machine where M is the model size. This would mean that for 70B we need 280GB of CPU on top of what we needed before (e.g. due to CPU offloading). This requirement is only for rank 0 though and not any other machine. So it's important to schedule this process on a machine with this much of RAM while the other processes can get scheduled on machines with lower RAM requirements. 

For example, for 70B model, with 32-way sharding on a machine with 8xA10Gs (g5.48xlarge), you need 280G (because of checkpointing) and 315 GB (because of optimizer state offloading) making the total memory requirement ~595 GB.

Ray provides an easy way to control which process gets launched on what machine type. To do this, in your cluster config add a custom lable for those machines that satisifies the CPU RAM requirement of rank 0 and call them `large_cpu_mem` instances. Then in our script we specify the custom tag as a resource requiremnet for the `trainer` actor which is in the same machine that rank zero process will get executed on.

```
scaling_config=air.ScalingConfig(
    # "large_cpu_mem" is the tag used to identify this machine type in the
    # cluster config.
    trainer_resources={"large_cpu_mem": 0.01},
    num_workers=args.num_devices,
    use_gpu=True,
    resources_per_worker={"GPU": 1},
)
```


### Submiting a production job
You can easily submit a production job using the following command:

```
python create_job_yaml.py --size=7b --output-path=./job.yaml
```

This will create a job yaml file that you can use to submit a production job on Anyscale platform. 

```
anyscale job submit job.yaml
```
