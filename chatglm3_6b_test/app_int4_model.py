from bigdl.llm.transformers import AutoModelForCausalLM
# 自动加载模型
from transformers import AutoTokenizer
# 自动模型张量
import time
import torch

tokenizer_path = './chatglm3_6b'
model_path = './chatglm3_6b_bigdl_llm_INT4'

model = AutoModelForCausalLM.load_low_bit(model_path,
                                          trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                          trust_remote_code=True)

CHATGLM_V2_PROMPT_TEMPLATE = "问：{prompt}\n\n答："

prompt = "鲁迅是周树人的哥哥吗？写个1000字说明"
n_predict = 3200

with torch.inference_mode():
    prompt = CHATGLM_V2_PROMPT_TEMPLATE.format(prompt=prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids,
                            max_new_tokens=n_predict)
    output_str = tokenizer.decode(output[0], skip_special_tokens=True)
    print('-'*20, 'Output', '-'*20)
    print(output_str)
