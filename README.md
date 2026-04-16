# TO DO make a nice README

# Mention the need for 3 different env !!! and explain which is what !!! 

# Explain the whole ordeal of ESM2 and only running it on C.Elegan / Drosophila / E.Coli 

## Environment installation SeqVec:

This is because SeqVec comes from Bio-embeddings which is an old package that is not longer maintained and has very specific requirements. 

```bash

# Or conda
mamba create -n bio_emb_recreated --file environment/seq_vec_emb_detailed_env.txt
 
python -m pip install -f https://download.pytorch.org/whl/cu113/torch_stable.html torch==1.10.0+cu113

python -m pip install "bio-embeddings[seqvec] @ git+https://github.com/sacdallago/bio_embeddings.git@develop"

```

## Environment installation ESM2:

## Changes to the orignal code of Graph-BERT: 

GPU selection + CPU limitation was added, these were changes in files script_1 / 2 / 3 and 4 in both the `Graph-BERT` and `Modiefied_Graph-BERT` folders. These changes are clearly specified in each of the files via a docstring that explains each individual change. Overall, the behavior of the code was not altered just adapted to suit a shared laboratory cluster.  

## Dataset:

Specify that the dataset is not uploaded here since it's too big for Githubs 100 Mb hardcap that is why it's in the gitignore. 

| Dataset | Link | `dataset_name` in code |
|---|---|---|
| Human PPI (HPRD benchmark) | [Google drive](https://drive.google.com/drive/folders/1KX9ybM_Mh2RXvqCmN7vJ2X_ywg12DJtq) | `ppi` |
| C.elegans | [Google drive](https://drive.google.com/drive/folders/1vNPpyGYDFHjHd2ylB77SmAwnh0tG18Vw) | `c.elegan` |
| Drosophila | [Google drive](https://drive.google.com/drive/folders/1_4oZZVvBFiub0FPOyQxVEmtZVM3kHqkQ) | `drosophila` |
| E. coli | [Google drive](https://drive.google.com/drive/folders/1wK6S_bZIDWaxW9IIU_GukyWD_ateqL2P) | `e.coli` |


But also put all the links and the architecture of where the data is meant to be situated so like 

```text
Graph-Bert or Modified_Graph-Bert/
├── data/
│   ├── ppi/
│   │   ├── node
│   │   └── link
│   ├── c.elegan/
│   │   ├── node
│   │   └── link
│   ├── drosophila/
│   │   ├── node
│   │   └── link
│   └── e.coli/
│       ├── node
│       └── link
```
