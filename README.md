# PCA Based Model Modification

## Dependencies

- python == 3.7
- pytorch == 1.12.1
- numpy == 1.21.5
- thop == 0.1.1

## Implementation


#### Compute_PCA and Extract Primary Layers

- Calculate PCA explained variance ratio from compressed output for each activation layer
- Please check out section 1 and 2 in [quantization_conversion.ipynb](./quantization_conversion.ipynb)

#### Training Script
        
- Currently available for Cifar10/Cifar100 training for vgg11/13/16/19
Details in [train.py](./train.py)
        
``` bash
        python train.py --pretrained --dataset cifar10 --model vgg13
```

#### quantization

- Please check out [quantization_conversion.ipynb](./quantization_conversion.ipynb)
Detailed instructions are to be added



