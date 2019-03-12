# Chinese QA Bot

Using [transformer](https://arxiv.org/abs/1706.03762) to train a Chinese chatbot.

## Dataset

- Reference: [PTT 中文語料](https://github.com/zake7749/Gossiping-Chinese-Corpus)
- Download dataset: `sh download_data.sh`

## Code Reference

- Google transformer architecture: [tensorflow/models/official/transformer](https://github.com/tensorflow/models/tree/master/official/transformer)

## Installation

`pip3 install -r requires.txt`

## Training pipeline

1. Build the data file with `.tfrecord` format:
```
python3 build_data.py
```

2. Train your model:
```
python3 train.py config/test_config.yml
```

You can customize your model architecture by writing a new `.yml` file. 
For more detail, see `config/test_config.yml`

If you want to change the learning rate, total training steps or other training strategies, please modify the code in `train.py`.

## Tensorboard

Type the following command and check the url: http://localhost:8080

```
tensorboard --logdir build --port 8080
```

![](https://github.com/st9007a/ChineseQABot/blob/master/image/tensorboard.PNG)

## Run a simple chatbot

`train.py` will export a [Tensorflow SavedModel](https://www.tensorflow.org/guide/saved_model#build_and_load_a_savedmodel) every 100000 training steps.
Those models will be placed under `serve` folder.

To run a simple demo, make sure SavedModel exist and type the following command:
```
python3 chat.py serve/[YOUR MODEL FOLDER]
```

![](https://github.com/st9007a/ChineseQABot/blob/master/image/chat.PNG)
