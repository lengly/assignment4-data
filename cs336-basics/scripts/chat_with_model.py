#!/usr/bin/env python3
"""
交互式聊天脚本，用于与训练好的语言模型对话
"""

import json
import torch
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from transformers import AutoTokenizer
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.train_config import Config

def load_model(model_path: str, device: str = "cuda:0"):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型权重目录路径
        device: 设备类型
    
    Returns:
        model: 加载好的模型
        tokenizer: GPT-2 tokenizer
    """
    print(f"正在加载模型从: {model_path}")
    
    # 加载模型配置
    config_path = os.path.join(model_path, "model_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"模型配置: {config}")
    
    # 使用现有的Config类创建配置对象
    cfg = Config()
    # 更新配置以匹配模型参数
    cfg.model.vocab_size = config['vocab_size']
    cfg.model.context_length = config['context_length']
    cfg.model.d_model = config['d_model']
    cfg.model.num_layers = config['num_layers']
    cfg.model.num_heads = config['num_heads']
    cfg.model.d_ff = config['d_ff']
    cfg.model.rope_theta = config['rope_theta']
    
    model = BasicsTransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=config['rope_theta'],
        cfg=cfg
    )
    
    # 加载权重 - 使用DeepSpeed ZeRO-2格式的权重文件
    weights_path = os.path.join(model_path, "mp_rank_00_model_states.pt")
    if os.path.exists(weights_path):
        print(f"加载权重文件: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # 提取模型状态字典
        if 'module' in checkpoint:
            state_dict = checkpoint['module']
        else:
            state_dict = checkpoint
        
        # 移除可能的DeepSpeed前缀
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                cleaned_key = key[7:]  # 移除 'module.' 前缀
            else:
                cleaned_key = key
            cleaned_state_dict[cleaned_key] = value
        
        # 加载状态字典
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        if missing_keys:
            print(f"警告: 缺少的键: {missing_keys}")
        if unexpected_keys:
            print(f"警告: 意外的键: {unexpected_keys}")
    else:
        print(f"错误: 找不到权重文件 {weights_path}")
        sys.exit(1)
    
    # 加载tokenizer
    print("加载GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 将模型移动到指定设备并设置为评估模式
    model = model.to(device)
    model.eval()
    
    # 设置模型为bfloat16以兼容FlashAttention
    if device.startswith("cuda"):
        model = model.bfloat16()
    
    print("模型加载完成!")
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 100, 
                     temperature: float = 0.7, top_k: int = 50, device: str = "cuda:0"):
    """
    生成模型响应
    
    Args:
        model: 语言模型
        tokenizer: tokenizer
        prompt: 输入提示
        max_new_tokens: 最大生成token数
        temperature: 温度参数
        top_k: top-k采样参数
        device: 设备类型
    
    Returns:
        generated_text: 生成的文本
    """
    # 编码输入
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        # 生成文本
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # 移除原始提示，只返回生成的部分
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text

def main():
    """主函数"""
    # 配置参数
    model_path = "/workspace/models"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    max_new_tokens = 200
    temperature = 0.7
    top_k = 50
    
    print("=" * 60)
    print("欢迎使用语言模型聊天系统!")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"最大生成token数: {max_new_tokens}")
    print(f"温度: {temperature}")
    print(f"Top-K: {top_k}")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'help' 查看帮助")
    print("=" * 60)
    
    try:
        # 加载模型
        model, tokenizer = load_model(model_path, device)
        
        # 交互式对话循环
        while True:
            try:
                # 获取用户输入
                user_input = input("\n你: ").strip()
                
                # 检查退出命令
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("再见!")
                    break
                
                # 检查帮助命令
                if user_input.lower() == 'help':
                    print("\n帮助信息:")
                    print("- 直接输入文本开始对话")
                    print("- 输入 'quit', 'exit' 或 'q' 退出")
                    print("- 输入 'help' 查看此帮助信息")
                    continue
                
                # 检查空输入
                if not user_input:
                    print("请输入一些内容...")
                    continue
                
                print("模型正在思考...")
                
                # 生成响应
                response = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=user_input,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    device=device
                )
                
                print(f"模型: {response}")
                
            except KeyboardInterrupt:
                print("\n\n再见!")
                break
            except Exception as e:
                print(f"生成过程中出现错误: {e}")
                continue
    
    except Exception as e:
        print(f"加载模型时出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
