import torch
from peft import LoraConfig

def add_lora_to_unet(unet, rank=4):
    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        check_flag = 0
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break
    unet.add_adapter(LoraConfig(r=rank,init_lora_weights="gaussian",target_modules=l_target_modules_encoder), adapter_name="default_encoder")
    unet.add_adapter(LoraConfig(r=rank,init_lora_weights="gaussian",target_modules=l_target_modules_decoder), adapter_name="default_decoder")
    unet.add_adapter(LoraConfig(r=rank,init_lora_weights="gaussian",target_modules=l_modules_others), adapter_name="default_others")
    return unet