import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    output_attentions=True,
    output_hidden_states=True
).to(device)

text = "The capital of France is"
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

print("Logits shape:", outputs.logits.shape)
print("Number of hidden states:", len(outputs.hidden_states))
print("Hidden state shape (layer 0):", outputs.hidden_states[0].shape)
print("Number of attention layers:", len(outputs.attentions))
print("Attention shape (layer 0):", outputs.attentions[0].shape)
