# Automatic-Opinion-Question-Generation

Implementation of Automatic Opinion Question Generation paper in [PyTorch](https://github.com/pytorch/pytorch).

Place your data in data/raw .

### Generate the processed data:
```
python file_convertion.py
```

### Preprocessing:
```
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/data -lower
```

### Generate embedding files:
```
python embedding.py -embedding  <path to embedding txt file> -dict data/data.{src,tgt}.dict -output data/{src,tgt}.840B.300d
```

### Training:
```
python train.py -data data/data.train.pt -save_model model/model -coverage_attn -attention_type general -lambda_coverage 1 -brnn -rnn_size 600 -word_vec_size 300 -epochs 20 -start_decay_at 10 -layers 2 -pre_word_vecs_enc data/src.840B.300d -pre_word_vecs_dec data/tgt.840B.300d
```
Use -gpus 0 if a GPU is available,o denotes gpuid.
Use nohup command if it takes lot of time to train

### Generating:
```
python translate.py -model model/model_epochX_PPL.pt -src data/src-test.txt -output result/pred.txt -replace_unk -verbose
```

### Evaluation: 
```
cd Evaluation
./eval.py --out_file ../result/pred.txt 
```

## Acknowledgment

Our implementation is adapted from [OpenNMT](http://opennmt.net). The evaluation scripts are adapted from [coco-caption](https://github.com/tylin/coco-caption) repo.

## License

Code is released under [the MIT license](http://opensource.org/licenses/MIT).
