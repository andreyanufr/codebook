import torch
import torch.nn.functional as F

logits = torch.randn(20, 32)
# Sample soft categorical using reparametrization trick:
res = F.gumbel_softmax(logits, tau=1, hard=False)
# Sample hard categorical using "Straight-through" trick:
res_hard = F.gumbel_softmax(logits, tau=1, hard=True)

print(res)
print(res_hard)
