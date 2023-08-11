# stablelm_lora

Worked with VRAM 8GB (RTX 3070Ti). 

## Setup

```bash
git clone https://github.com/p1atdev/stablelm_lora
cd stablelm_lora
pip install -r requirements.txt
```

### Install bitsandbytes 

#### for Linux

(not tested)

```bash
pip install bitsandbytes
```

#### for Windows

See https://github.com/jllllll/bitsandbytes-windows-webui.

```bash
pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
```

## Train

**Please edit `example_train.py` to match your dataset and settings.**

Then,

```bash
python ./example_train.py
```

to start training.

## References


- [japanese-stablelm-base-alpha-7bのLoRAを試す](https://zenn.dev/saldra/articles/87d3b289583a68)
- [Windows10でのPEFTの実行方法 (bitsandbytes-windows-webui使用)](https://qiita.com/selllous/items/c4880767da45173227c1)

