Specify that the dataset is not uploaded here since it's too big for Githubs 100 Mb hardcap that is why it's in the gitignore. 

| Dataset | Link | `dataset_name` in code |
|---|---|---|
| Human PPI (HPRD benchmark) | [Google drive](https://drive.google.com/drive/folders/1KX9ybM_Mh2RXvqCmN7vJ2X_ywg12DJtq) | `ppi` |
| C.elegans | [Google drive](https://drive.google.com/drive/folders/1vNPpyGYDFHjHd2ylB77SmAwnh0tG18Vw) | `c.elegan` |
| Drosophila | [Google drive](https://drive.google.com/drive/folders/1_4oZZVvBFiub0FPOyQxVEmtZVM3kHqkQ) | `drosophila` |
| E. coli | [Google drive](https://drive.google.com/drive/folders/1wK6S_bZIDWaxW9IIU_GukyWD_ateqL2P) | `e.coli` |


But also put all the links and the architecture of where the data is meant to be situated so like 

Graph-Bert/
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
