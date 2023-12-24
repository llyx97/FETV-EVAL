# convert json file with a single line to multiple lines

import json

orig_file = 'msrvtt_ret_train7k_neg_caption'
tgt_file = orig_file+'_lines'

with open(orig_file+'.json', 'r') as f:
    datas = json.load(f)

with open(tgt_file+'.json', 'w') as f:
    for d in datas:
        dumped = json.dumps(d)
        f.write(dumped+'\n')
