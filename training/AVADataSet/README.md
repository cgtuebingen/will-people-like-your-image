## AVADataSet

### Description:


The create_ava_lmbd.py script creates a lmdb file for the AVA-Dataset when given the AVA.txt and the images of the AVA-Dataset.The lmdb-File can either store the mean of the given valuations or the distribution of the given valuations.
### Usage create_ava_lmbd.py
```
python create_ava_lmbd.py --lmdbFile (path to lmdb file) --images ( path to AVA images) --avaTxt (path to AVA.txt) [--distribution (store the distribution of the valuations instead of the mean)]
```

The create_ava_lmbd.py is for testing purposes and displays the images contained in the lmdb-File with their mean-score or their distribution.
### Usage read_ava_lmdb.py
```
python read_ava_lmdb.py --lmdbFile (path to lmdb file) [--distribution (loaded lmdb is a distribution)] 

```

The validate_ava_lmdb.py compares a given lmdb-File with the avaTxt. It compares the scores or distributions as well as the completeness of the lmdb and prints out missing image names. To use the validate_ava_lmdb script the lmdb needs to additionally store the names of the images between the image itself and the score or distribution.
### Usage validate_ava_lmdb.py
```
python validate_ava_lmdb.py --lmdbFile (path to lmdb file) --avaTxt (path to AVA.txt) [--distribution (loaded lmdb is a distribution)] 

```

The ava_provider.py script contains a class named LabeledImage which can be used to load an image from the lmdb-File with its score for a training process.
### Usage ava_provider.py
```
python ava_provider.py --lmdbFile (path to lmdb file) [--distribution (loaded lmdb is a distribution)] 
```

The alex_net_for_classification_learning.py script performs a classification learning on
The script is only capable of training the mean score.
### Usage alex_net_for_classification_learning.py
```
python alex_net_for_classification_learning.py --gpu (gpu id) --data (path to lmdb file)
```
