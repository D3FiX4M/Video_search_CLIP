import torch

model = torch.jit.load('lib/model.pt')
model.cuda().eval()

input_resolution = model.input_resolution.item()
context_length = model.context_length.item()
vocab_size = model.vocab_size.item()
