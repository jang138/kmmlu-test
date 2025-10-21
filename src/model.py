# src/model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model_and_tokenizer(model_name: str, device: str = "cpu"):
    """모델과 토크나이저 로드"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.to(device)

    return model, tokenizer


if __name__ == "__main__":
    model_name = "skt/kogpt2-base-v2"
    model, tokenizer = load_model_and_tokenizer(model_name)
    print(f"Model loaded: {model.config.model_type}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
