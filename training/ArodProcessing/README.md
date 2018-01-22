## ArodProcessing

### Description:
This directory contains all necessary files to use the AROD-Dataset for the learning of embedding.


The arod_provider-py script  contains a class named Triplets which can be used to generate appropriate triplets from the AROD-Dataset.
### Usage arod_provider.py 
```
python arod_provider.py --lmdb(path to lmdb file)
```


The triplets_to_txt_file.py script creates with the help of the mentioned Triplet class a text file containing the ids of the created Triplets.
### Usage triplets_to_txt_file.py
```
python triplets_to_txt_file.py --data (path to lmdb file) --outFile (path to wished txt file)
```



The arod_dataflow_from_txt.py script also contains a class named Triplets which also can be used as the input for the training process. This class evaluates a given text file which stores triplets for generating the needed triplets for training.
### Usage arod_dataflow_from_txt.py 
```
python arod_dataflow_from_txt.py --lmdbFile (path to lmdb file) --tripletFile (path to txt file with stored triplets)

```
