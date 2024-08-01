# 使用pytorch/pytorch：2.1.2-cuda12.1-cudnn8-devel作为基础镜像

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

RUN apt-get update -y --allow-unauthenticated
RUN apt install -y git vim git-lfs

# 设置工作目录
WORKDIR /root/llama-finetune-lora

# 开启7860端口
EXPOSE 7860

# 设置启动命令
ENTRYPOINT [ "python", "deploy/chat_gradio.py", "--model_name_or_path", "/root/llama3-dev/train/stf/lora_finetune_llama3/checkpoint-12300" ]