# DARTS_with_Post_Training_Quantization

## Running post-training quantization experiments:

### Step 1: clone this repo
`git clone https://github.com/udday2014/DARTS_with_Post_Training_Quantization/`

### Step 2: cd into the cnn directory
`cd cnn`

### Step 3: run the command (baseline):
`!python test_modified.py --auxiliary --do_quant 0 --model_path cifar10_model.pt`
here, do_quant parameter defines whether we want to perform quantization or not. by default its 0 (means no quantization)

### Step 4: run the command (with quant):
`python test_modified.py --auxiliary --do_quant 1 --model_path cifar10_model.pt --param_bits 8 --fwd_bits 8 --n_sample 10`

### Quantization parameters:
These can be found in the **test_modified.py** script:\
**quant_method:**, quantization function: possible choice: linear|minmax|log|tanh\
**param_bits:** bit-width for parameters\
**bn_bits:** bit-width for running mean and std in batchnorm layer (should be higher)\
**fwd_bits:** bit-width for layer output (activation)\
**n_sample:** number of samples to calculate the scaling factor

### Post-quantization result (CIFAR10)

| Weight-bit    | Activation-bit| Acc   |
| :-----------: |:-------------:| -----:|
| 32      `     | 32            | 97.37 |
| 8             | 8             | 97.44 |
| 4             | 8             | 88.87 |
| 4             | 4             | 18.02 |
