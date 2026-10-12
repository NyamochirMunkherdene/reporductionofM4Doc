from transformers import AutoTokenizer, AutoModelForCausalLM

path = "/home/pj24001974/ku60000761/project/test_project/vary-toy"

tokenizer = AutoTokenizer.from_pretrained(
    path,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map="cpu",
)

print(type(model))
print(model)
print("has transformer:", hasattr(model, "transformer"))