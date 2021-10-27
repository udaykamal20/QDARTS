#PBS -N pytorchTest
#PBS -A mjurado3@gatech.edu
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l pmem=2gb
#PBS -l walltime=10:00
#PBS -q coc-ice-gpu
#PBS -j oe
#PBS -o pytorchTest.out

cd /storage/home/hcocice1/mjurado3/darts_with_mips/DARTS_with_Post_Training_Quantization/cnn
# module load pytorch/1
module load pytorch
#module load anaconda3
#conda activate darts

python test_modified.py --auxiliary --do_quant 1 --model_path cifar10_model.pt  --param_bits 8 --fwd_bits 8 --n_sample 10
