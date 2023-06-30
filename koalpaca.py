import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import PeftModel, PeftConfig
from peft import prepare_model_for_kbit_training

import os, sys, time
from datetime import datetime

import torch, gc
gc.collect()
torch.cuda.empty_cache()
#del variables
# print(torch.cuda.memory_summary(device=None, abbreviated=False))
model = None
model_type = "koalpaca-355m"
# model_type = "128B-GPTQ"
# model_type = "KoRWKV-6B"
# model_type = "KoRWKV-1.5B"
# model_type = "58b"
# model_type = "128b"
# model_type = "128b-4bit-quant"


if model_type == "128b-4bit-quant":
    # 12.8b, 4bit quantized model
    peft_model_id = "beomi/qlora-koalpaca-polyglot-12.8b-50step"
    config = PeftConfig.from_pretrained(peft_model_id)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config, device_map={"":0})
    model = PeftModel.from_pretrained(model, peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    
elif model_type == "58b":
    # 5.8b
    model_id = "beomi/KoAlpaca-Polyglot-5.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"":0})
    
elif model_type == "koalpaca-355m":
    # 아주 좋은 성능은 아니지만 cpu 추론 가능함, 적은 메모리 사용
    model_id = "heegyu/koalpaca-355m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"":0})
    
elif model_type == "128B-GPTQ":
    tokenizer = AutoTokenizer.from_pretrained("../model/KoAlpaca-Polyglot-12.8B-GPTQ", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("../model/KoAlpaca-Polyglot-12.8B-GPTQ/", device_map={"":0}, local_files_only=True)
    
elif model_type == "KoRWKV-6B":
    tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-KoRWKV-6B")
    model = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-KoRWKV-6B", device_map={"":0})
    
elif model_type == "KoRWKV-1.5B":
    tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-KoRWKV-1.5B")
    model = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-KoRWKV-1.5B", device_map={"":0})
    
elif model_type == "128B":
    tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-12.8B")
    model = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-Polyglot-12.8B", device_map={"":0})
    
if model:
    model.eval()
    model.config.use_cache = True
else:
    sys.exit(1)

def gen(x):
    start_time = datetime.now()
    
    q = f"### 질문: {x}\n\n### 답변:"
    # print(q)
    # gened = model.generate(
    #     **tokenizer(
    #         q,
    #         return_tensors='pt',
    #         return_token_type_ids=False
    #     ).to('cpu'),
    #     max_new_tokens=250,
    #     early_stopping=True,
    #     do_sample=True,
    #     eos_token_id=2,
    #     num_return_sequences=1,
    #     temperature=0.7,
    #     top_p=0.95,
    #     repetition_penalty=1.2,
    # )
    
    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        max_new_tokens=250,
        do_sample=True,
        eos_token_id=2,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.2,
    )
    result = tokenizer.decode(gened[0], skip_special_tokens=True)
    elapsed_time = datetime.now() - start_time
    # print(f"result: {result}")
    print(f"elapsed time for generating message: {elapsed_time}\n\n")
    return result
    