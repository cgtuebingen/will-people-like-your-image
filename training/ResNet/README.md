## ResNet

### Description:

The npz_file_to_checkpoint.py script tries to convert the content of a npz directory to a tensorflow checkpoint. With this script
pretrained weights from earlier training processes can be reused.
### Usage npz_file_to_checkpoint.py
```
python npz_file_to_checkpoint.py --npz (path to .npz folder) --model (path of the wished pretrained model) --target (location of the used target model for comparison)
```

The resnetXX_for_embedding.py scripts implement the training of the embedding according to the presented method.
The training is based on a residual network architecture.

### Usage resnet18_for_embedding.py
```
python resnet18_for_embedding.py --gpu (gpu ids) --lmdbFile (path to lmdb file) --tripletFile (path to txt file with stored triplets) --load (path to tensorflow checkpoint with pretrained model)
```

### Usage resnet34_for_embedding.py
```
python resnet34_for_embedding.py --gpu (gpu ids) --lmdbFile (path to lmdb file) --tripletFile (path to txt file with stored triplets) --load (path to tensorflow checkpoint with pretrained model)
```

### Usage resnet50_for_embedding.py
```
python resnet50_for_embedding.py --gpu (gpu ids) --lmdbFile (path to lmdb file) --tripletFile (path to txt file with stored triplets) --load (path to tensorflow checkpoint with pretrained model)
```
