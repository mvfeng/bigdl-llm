from bigdl.llm.transformers import AutoModelForCausalLM

model_path = './chatglm3_6b'

model = AutoModelForCausalLM.from_pretrained(model_path,
                                             load_in_4bit=True,
                                             trust_remote_code=True)
save_directory = './chatglm3_6b_bigdl_llm_INT4'

model.save_low_bit(save_directory)

