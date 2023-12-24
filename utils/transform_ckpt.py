import torch, collections

ckpt_name = 'pytorch_model.bin.4'
state_dict = torch.load(ckpt_name, map_location="cpu")
new_state_dict = collections.OrderedDict()
for key, val in state_dict.items():
    new_key = key.replace('clip.', '')
    new_state_dict[new_key] = val.clone()
torch.save(new_state_dict, 'ViT-B-32-epoch4.bin')
