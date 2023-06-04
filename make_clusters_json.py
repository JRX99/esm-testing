NUMBER_OF_CLUSTERS=30


#important imports
import pandas as pd
import numpy as np
import os
import json
import pathlib
import json
from Bio import SeqIO



home_folder=pathlib.Path(__file__).parent.resolve()
output_folder=os.path.join(home_folder,"clusters")
input_file=os.path.join(home_folder,"uniprot","uniref50.fasta")
chunk_size=3 #should be divideable by number of clusters

cluster_id=1

with open(input_file, mode='r') as handle:
    for record in SeqIO.parse(handle, 'fasta'):
        desc=record.description.split(" ")
        for x in desc:
            if "TaxID=" in x:
                organism_id=x.split("=")[1]
            if "RepID=" in x:
                protein_id=x.split("=")[1]
        
        protein_sequence=str(record.seq)
            
        f = open(os.path.join(output_folder,"cluster"+str(cluster_id)+".tsv"), "a")
        f.write(str(protein_id)+" \t "+str(organism_id)+" \t "+str(protein_sequence)+"\n")
        f.close()

        if cluster_id==NUMBER_OF_CLUSTERS: cluster_id=0
        cluster_id+=1
    