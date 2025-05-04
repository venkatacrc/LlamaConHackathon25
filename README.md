# LlamaConHackathon25
## Distilling the Llama4 Scout model

### Setup
Launch a 8x H100 GPU cluster gpu_8x_h100_sxm5 

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda create --name llamaenv python=3.10
conda activate llamaenv

sudo apt install python3-pip
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install torchtune
git config --global credential.helper store
huggingface-cli login

### Download the Model Weights
tune download meta-llama/Llama-4-Scout-17B-16E-Instruct --hf-token <HF_TOKEN>

### Distill the llama4 model to a smaller model
tune run --nproc_per_node 8 knowledge_distillation_distributed --config recipes/configs/llama4/llama4_to_llama3_1B_KD_lora_distributed

### Evaluating the model
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness/
pip install -e .
python -m lm_eval --model hf --model_args pretrained=/tmp/torchtune/llama4_scout_to_1B/KD_lora_distributed/epoch_0   --tasks truthfulqa_mc2,hellaswag,commonsense_qa   --device cuda

### Issues
In this tutorial
https://pytorch.org/torchtune/stable/tutorials/llama_kd_tutorial.html
this command for distillation needs to be updated from:
```tune run knowledge_distillation_single_device --config llama3_2/knowledge_distillation_single_device```
to:
```tune run knowledge_distillation_single_device --config llama3_2/8B_to_1B_KD_lora_single_device```

