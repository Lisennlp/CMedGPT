# CMedGpt

## Description
  中文医疗GPT

## train
bash scripts/run.sh train

## eval
bash scripts/run.sh eval

## 如果你想使用fp16训练
###  安装apex
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install

### apex问题
如果提示：
if cached_x.grad_fn.next_functions[1][0].variable is not x: # error line
IndexError: tuple index out of range
则请看：
https://github.com/NVIDIA/apex/pull/1282