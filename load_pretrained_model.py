from transformers import GPT2LMHeadModel
from torch.nn import functional as F
import tiktoken
import torch

torch.manual_seed(42)
torch.cuda.manual_seed(42)

def generate(model, encoding, prefix, max_length=30, num_samples=5, topk=50):
    input = torch.tensor(encoding.encode(prefix), dtype=torch.long)
    input = input.unsqueeze(0)
    # B x T
    input = input.repeat(num_samples, 1)

    while input.size(1) < max_length:
        # B x T x C
        logits = model(input).logits
        # B x C
        probs = F.softmax(logits[:, -1, :])
        # B x topK
        topk_probs, topk_indices = torch.topk(probs, topk, dim=-1)
        # B x 1
        next_token_index = torch.multinomial(topk_probs, 1)
        # B x 1
        gathered_indices = torch.gather(topk_indices, -1, next_token_index)
        input = torch.cat((input, gathered_indices), dim=-1)

    return input


if __name__ == "__main__":
    enc = tiktoken.get_encoding("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2") # 124M
    model.eval()
    
    text = "Hello, I'm a language model,"
    generated = generate(model, enc, text, max_length=30, num_samples=5) 
    for gen_text in generated.tolist():
        print(enc.decode(gen_text))