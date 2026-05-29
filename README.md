# Methods in Bioinformatics (INFO-F439)

The article that I have chosen from this course stems from K. Jha, S. Karmakar, and S. Saha. “Graph-BERT and language model-based framework for protein–protein interaction identification”. en. In: Scientific Reports 13.1 (Apr. 2023), p. 5663. doi: 10.1038/s41598-023-31612-w. 

## Environment installation SeqVec:

This is because SeqVec comes from Bio-embeddings which is an old package that is not longer maintained and has very specific requirements. 

```bash

# Or conda
mamba create -n seq_vec_emb --file environment/seq_vec_emb_detailed_env.txt

mamba activate seq_vec_emb

python -m pip install -f https://download.pytorch.org/whl/cu113/torch_stable.html torch==1.10.0+cu113

python -m pip install "bio-embeddings[seqvec] @ git+https://github.com/sacdallago/bio_embeddings.git@develop"

```

## Environment installation ESM2:

```bash

# Or conda
mamba create -n esm_emb python=3.10 pip -y

mamba activate esm_emb

pip install torch --index-url https://download.pytorch.org/whl/cu118

pip install transformers accelerate
```

## Changes to the orignal code of Graph-BERT: 

Some changes were made in files script_1 / 2 / 3 and 4 in both the `Graph-BERT` folder. These changes are clearly specified in each of the files via a docstring that explains each individual change. Overall, the behavior of the code was not altered just adapted to suit a shared laboratory cluster, ultimately these concern aspects such as GPU selection with CPU limitation with a more appropriate chosing of which dataset it meant to be used without using hardcoded variables. 

## Dataset and SeqVec / ESM-2 weights:

Concerning the original and produced datasets, and especially ESM-2 weights these are all present on this Google drive [https://drive.google.com/drive/folders/1ebuwW51Fo05qFbxVlUjVwS31oRxY5D06?usp=sharing]. This is due to GitHub's size limit for pushes and individual files. The specified google drive possess five different folders: 

1. `data` which contains the original data used by the authors, this same folder is also present in the `Graph-BERT` folder on this repository
2. `esm-files`, which contains the ESM-2 dictionaries for the used datasets in the 650M and 3B parameter settings
3. `Node_creation`, which contains the node files created for this project
4. `S-VGAE` data which is the original data from which the authors have created their own node and link files, this was used in this project for the re-creation of the node files
5. `seqvec_files`, which contain the SeqVec weight dictionaries that were produced for this project 


## How to run the repository:


```bash
# Step 1 

# To be ran in Methods/

# C.elegan
python pre-processing/generate_node_v2.py --dataset c.elegan --embedder seqvec
python pre-processing/generate_node_v2.py --dataset c.elegan --embedder esm2_650M
python pre-processing/generate_node_v2.py --dataset c.elegan --embedder esm2_3B

# E.coli
python pre-processing/generate_node_v2.py --dataset e.coli --embedder seqvec
python pre-processing/generate_node_v2.py --dataset e.coli --embedder esm2_650M
python pre-processing/generate_node_v2.py --dataset e.coli --embedder esm2_3B

# Drosophila
python pre-processing/generate_node_v2.py --dataset drosophila --embedder seqvec
python pre-processing/generate_node_v2.py --dataset drosophila --embedder esm2_650M
python pre-processing/generate_node_v2.py --dataset drosophila --embedder esm2_3B

# Step 2 

# To be ran in Methods/Graph-BERT-ESM2/

./run.sh 8 0 script_1_preprocess.py c.elegan seqvec
./run.sh 8 0 script_1_preprocess.py e.coli seqvec
./run.sh 8 0 script_1_preprocess.py drosophila seqvec

# Step 3 

# To be ran in Methods/Graph-BERT-ESM2/

for emb in seqvec esm2_650M esm2_3B; do
    ./run.sh 8 0 script_2_pre_train.py  c.elegan "$emb"
    ./run.sh 8 0 script_3_fine_tuning.py c.elegan "$emb"
done

for emb in seqvec esm2_650M esm2_3B; do
    ./run.sh 8 0 script_2_pre_train.py  e.coli "$emb"
    ./run.sh 8 0 script_3_fine_tuning.py e.coli "$emb"
done

for emb in seqvec esm2_650M esm2_3B; do
    ./run.sh 8 0 script_2_pre_train.py  drosophila "$emb"
    ./run.sh 8 0 script_3_fine_tuning.py drosophila "$emb"
done


# Step 4 

# To be ran in Methods/Graph-BERT-ESM2

python script_4_evaluation_plots.py --dataset all --embedder all

```

