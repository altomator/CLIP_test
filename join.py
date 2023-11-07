# Python program to join tensors in PyTorch
# import necessary library
import torch
import os
import argparse
import glob

# folder where the tensors are
torchFolder = "torch/"

###
parser = argparse.ArgumentParser(
                    prog = 'join_tensors.py',
                    description = 'Concatenate the tensors for a use case'
                    )
parser.add_argument('-n', '-name', help='the name of the use case',required=True)
args = parser.parse_args()

use_case = args.n

print('...working on use case: '+ use_case )
print ('...looking for pattern: ' + torchFolder + use_case + "_torch*.pt")
torch_files = glob.glob(torchFolder + use_case + '_torch*.pt')

print(torch_files)

tensors=[]
for t_f in (torch_files):
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
output = torchFolder + use_case +  "_torch.pt"
print ("--> tensor saved in %s\n" % output)
torch.save(T, output)
