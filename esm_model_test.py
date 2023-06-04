import torch
import torch.hub
import pathlib
import os
import random
import copy
import numpy as np

MAX_LINES=10000
BATCH_SIZE=10


def batch_generator(file,batch_size=5):
    while True:
        data = ''.join(f.readline() for _ in range(batch_size))
        data = data.split("\n")
        for i in range(batch_size):
            data_segment = data[i].split("\t")
            sequence=data_segment[2]
            if len(sequence)>1000:
                data[i]=sequence[0:999].strip()
            else:
                data[i]=sequence.strip()
        if not data:
            break
        yield data



def mask_data(data):
    out_list=copy.deepcopy(data)
    probability=0.05 # 5% for sequence element to be masked

    for x in range(len(data)):
        temp = list(data[x])
        for i in range(0, len(temp)):
            if random.random() < probability:
                temp[i] = "<mask>"
        out_list[x] =tuple(("protein"+str(x),"".join(temp)))

    return out_list


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t6_8M_UR50D")
#model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t12_35M_UR50D")
model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t30_150M_UR50D")



model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results
file=os.path.join(os.getcwd(),"clusters","cluster1.tsv")

number_of_masks=0
number_of_correct_predictions=0
lines_read=0
with open(file,mode='r') as f:
    
    
    
    generator=batch_generator(f,batch_size=BATCH_SIZE)
    while lines_read<MAX_LINES:
        lines_read+=BATCH_SIZE
        data=next(generator)
        masked_data=mask_data(data)
        for x in range(len(data)):
             data[x] =tuple(("protein"+str(x),data[x]))



        batch_labels, batch_strs, batch_tokens_masked = batch_converter(masked_data)
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_tokens_masked=batch_tokens_masked.to(device)



        torch.cuda.empty_cache()
        with torch.no_grad():
            results = model(batch_tokens_masked)
        batch_tokens=batch_tokens.numpy()
        batch_tokens_masked=batch_tokens_masked.to('cpu')
        batch_tokens_masked=batch_tokens_masked.numpy()
        results=results['logits'].to('cpu')
        results=results.numpy()
        for i in range(batch_tokens.shape[0]):
            for s in range(len(batch_tokens_masked[i])):
                if batch_tokens_masked[i][s]==32:
                    target=batch_tokens[i][s]
                    prediction=np.argmax(results[i][s])
                    number_of_masks+=1
                    if target==prediction: number_of_correct_predictions+=1
        if number_of_masks!=0:
            print("Accuracy is "+str(number_of_correct_predictions/number_of_masks))
        else:
            print("Zero masks")   
        print("Lines read "+str(lines_read))
f = open("esm2_t30_150M_UR50D.txt", "a")
f.write("Accuraccy of esm2_t30_150M_UR50D model is "+str(number_of_correct_predictions/number_of_masks))


