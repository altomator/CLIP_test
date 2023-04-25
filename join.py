# Python program to join tensors in PyTorch
# import necessary library
import torch
import os
import argparse


###
parser = argparse.ArgumentParser(
                    prog = 'join_tensors.py',
                    description = 'Concatenate the tensors for a use case'
                    )
parser.add_argument('-n', '-name', help='the name of the use case',required=True)
args = parser.parse_args()

use_case = args.n

print('...working on use case: '+ use_case )

tensor_file1 = use_case + "_torch1.pt"
tensor_file2 = use_case + "_torch2.pt"
tensors=[]

for t_f in (tensor_file1, tensor_file2):
    if not os.path.exists(t_f):
        print("### Torch tensor is missing: "+t_f)
        quit()
    print("...loading the embeddings from: "+t_f)
    features = torch.load(t_f)
    tensors.append(features)
    # join (concatenate) above tensors using torch.cat()
    T = torch.cat(tensors)

print("T: ",T)
print("Tensor size: ", T.size())
output = use_case +  "_torch.pt"
print ("--> tensor saved in %s\n" % output)
torch.save(T, output)
