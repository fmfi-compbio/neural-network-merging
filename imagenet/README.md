# Preliminaries

Install pytorch and [timm](https://github.com/rwightman/pytorch-image-models).

# Training teachers

`bash distibuted_train.sh 2 <path to imagenet> --config config_teacher1.yaml`

`bash distibuted_train.sh 2 <path to imagenet> --config config_teacher2.yaml`

# Training students

Copy results of training into `merging/teachers/` under names of `t1.pth.tar` and 
`t2.pth.tar`.

Then:
```
cd merging
mkdir students
bash distributed_train.sh 2 <path to imagenet> --config config_big_student.yaml
python go_back.py students/out11avg.pt students/compressed.pt
```

# Finetuning

```
cd ..
bash distributed_train.sh 2 <path to imagenet> --config config_finetune.yaml --initial-checkpoint merging/students/compressed.pt
```

# Validating

```
python validate.py <path to imagenet>/val --model resnet18 --checkpoint
<checkpoint path>
```
