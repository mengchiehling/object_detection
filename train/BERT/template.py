"""
Script for transfer learning with BERT model with hyperparameter optimization with Bayesian optimization

command line: python -m train.models._relevance_bert_v2 --model_type=<str> --model_name=<str>
--num_labels=<int> --optimizer=<str> --use_cuda=<boolean> --balanced=<boolean>
"""

import argparse
import shutil
from typing import Union

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from scipy.special import softmax
from simpletransformers.classification import ClassificationModel

from train.utils.utils import production_training_data_preparation, optimization_process
from review_filter.settings import DIR_TRAINED_MODELS

"""
BERT german pretrained models:
1. distilbert-base-german-cased
   6-layer, 768-hidden, 12-heads, 66M parameters
   The German DistilBERT model distilled from the German DBMDZ BERT model bert-base-german-dbmdz-cased checkpoint.
2. bert-base-german-cased
   12-layer, 768-hidden, 12-heads, 110M parameters.
   Trained on cased German text by Deepset.ai
3. bert-base-german-dbmdz-cased
   12-layer, 768-hidden, 12-heads, 110M parameters.
   Trained on cased German text by DBMDZ
4. bert-base-german-dbmdz-uncased
   12-layer, 768-hidden, 12-heads, 110M parameters.
   Trained on uncased German text by DBMDZ

XLM german pretrained models:
1. xlm-mlm-ende-1024
   6-layer, 1024-hidden, 8-heads
   XLM English-German model trained on the concatenation of English and German wikipedia
2. xlm-clm-ende-1024
   6-layer, 1024-hidden, 8-heads
   XLM English-German model trained with CLM (Causal Language Modeling) on the concatenation of English and German wikipedia
"""


def bert_training_process(learning_rate: float = 0.00005, max_seq_length: Union[float, int] = 128,
                          num_train_epochs: Union[float, int] = 20, max_grad_norm: float = 1,
                          adafactor_beta1: float = 0, adafactor_clip_threshold: float = 1,
                          adafactor_decay_rate: float = -0.8, weight_decay: float = 0, adam_epsilon: float = 1e-8,
                          warmup_ratio: float = 0.06, gradient_accumulation_steps: Union[float, int] = 1,
                          model_stacking: bool = False):

    """
    BERT model training with hyperparameter tuning.

    Args:
        learning_rate:
            Optional; the learning rate for training
        max_seq_length:
            Optional; maximum sequence length the model will support (upper limit 512)
        num_train_epochs:
            Optional; the number of epochs the model will be trained for.
        max_grad_norm:
            Optional; maximum gradient clipping.
        adafactor_beta1:
            Optional; coefficient used for computing running averages of gradient.
        adafactor_clip_threshold:
            Optional; coefficient used for computing running averages of gradient.
        adafactor_decay_rate:
            Optionall; coefficient used to compute running averages of square.
        weight_decay:
            Optional; adds L2 penalty.
        adam_epsilon:
            Optional; epsilon hyperparameter used in AdamOptimizer.
        warmup_ratio:
            Optional; ratio of total training steps where learning rate will “warm up”.
        gradient_accumulation_steps:
            Optional; the number of training steps to execute before performing a
            optimizer.step(). Effectively increases the training batch size while sacrificing training time to lower
            memory consumption.
        model_stacking:
            Optional; if returning out-of-fold predictions for model stacking
    Returns:
        1. if not model_stacking float, metric evaluation
        2. if model_stacking, models (dictionary) and a oof prediction (np.ndarray)
    """

    model_args = {'evaluate_during_training': False,
                  'num_train_epochs': int(num_train_epochs),
                  'manual_seed': 42,
                  'learning_rate': learning_rate,
                  'train_batch_size': 16,
                  'save_model_every_epoch': False,
                  'save_steps': -1,
                  'sliding_window': True,
                  'max_seq_length': int(max_seq_length),
                  'optimizer': args.OPTIMIZER,
                  'weight_decay': int(weight_decay),
                  "warmup_ratio": warmup_ratio,
                  "gradient_accumulation_steps": int(gradient_accumulation_steps),
                  'silent': True
                  }

    if args.OPTIMIZER == 'Adafactor':
        model_args.update({"adafactor_beta1": adafactor_beta1,
                           "adafactor_clip_threshold": adafactor_clip_threshold,
                           "adafactor_decay_rate": adafactor_decay_rate,
                           'adafactor_relative_step': False,
                           "adafactor_warmup_init": False})
    else:
        model_args.update({'adam_epsilon': adam_epsilon,
                           'max_grad_norm': max_grad_norm})

    model_dict = {}

    df = X.copy()

    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    y_true = np.zeros(len(df))
    y_pred = np.zeros(len(df))
    y_oof = np.zeros(len(df))

    fold = 0

    for idx, (train_index, val_index) in enumerate(kf.split(df, df['labels'])):

        output_dir = f"{DIR_TRAINED_MODELS}/cv/output_dir_{idx + 1}"
        model_args.update({"output_dir": output_dir})

        df_train = df.iloc[train_index]
        df_val = df.iloc[val_index]
        y_train = df.iloc[train_index]['labels'].values.astype(int)
        y_val = df.iloc[val_index]['labels'].values.astype(int)

        if args.BALANCED:
            class_weight_vect = list(compute_class_weight('balanced', np.array([0, 1]), y_train))
        else:
            class_weight_vect = None

        model = ClassificationModel(model_type=args.MODEL_TYPE, model_name=args.MODEL_NAME,
                                    num_labels=args.NUM_LABELS, use_cuda=args.USE_CUDA, weight=class_weight_vect,
                                    args=model_args)

        model.train_model(df_train)

        predictions, raw_outputs = model.predict(df_val['text'].tolist())

        fold += 1
        model_dict[fold] = model

        y_true[val_index] = y_val
        y_pred[val_index] = predictions
        y_oof[val_index] = np.array([softmax(raw_output, axis=1)[:, 1].mean() for raw_output in raw_outputs])

    shutil.rmtree(f"{DIR_TRAINED_MODELS}/cv")

    f1 = f1_score(y_true, y_pred)

    if not model_stacking:
        return f1
    else:
        np.save(f'{DIR_TRAINED_MODELS}/y_bert.npy', y_oof)
        return model_dict, y_pred


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', dest="MODEL_TYPE", type=str, default='distilbert', help='type of transformers, '
                        'bert, distilbert, or xml for German pretrained. Supported model_type: '
                        'https://simpletransformers.ai/docs/classification-specifics/#supported-model-types')
    parser.add_argument('--model_name', dest="MODEL_NAME", default='distilbert-base-german-cased', type=str,
                        help='pretrained weight')
    parser.add_argument('--num_labels', dest='NUM_LABELS', type=int, help='number of classes for classification')
    parser.add_argument('--optimizer', dest='OPTIMIZER', type=str, help='AdamW(default), Adafactor')
    parser.add_argument('--balanced', dest="BALANCED", default=False, action='store_true', help='weight balancing')

    args = parser.parse_args()

    pbounds = {'learning_rate': (0.00001, 0.0001),
               'max_seq_length': (64, 256),
               'num_train_epochs': (4, 9),
               # 'adafactor_beta1': (0, 3),
               # 'adafactor_clip_threshold': (0.5, 1),
               # 'adafactor_decay_rate': (-1, -0.3),
               'adam_epsilon': (1e-8, 1e-6),
               'max_grad_norm': (0.5, 1),
               'weight_decay': (0, 10),
               'warmup_ratio': (0.03, 0.2)
               }
    X = production_training_data_preparation()

    optimized_parameters, y_oof = optimization_process(bert_training_process, pbounds)

    model_args = optimized_parameters

    model_args['num_train_epochs'] = int(model_args['num_train_epochs'])
    model_args['max_seq_length'] = int(model_args['max_seq_length'])
    model_args['weight_decay'] = int(model_args['weight_decay'])

    if args.BALANCED:
        class_weight_vect = list(compute_class_weight('balanced', np.array([0, 1]), X['labels'].values.astype(int)))
    else:
        class_weight_vect = None

    model_args.update({"optimizer": args.OPTIMIZER,
                       'evaluate_during_training': False,
                       'save_model_every_epoch': False,
                       'save_steps': -1,
                       'sliding_window': True,
                       'output_dir': f"{DIR_TRAINED_MODELS}/output_dir"})

    model = ClassificationModel(model_type=args.MODEL_TYPE, model_name=args.MODEL_NAME,
                                num_labels=args.NUM_LABELS, use_cuda=True, weight=class_weight_vect,
                                args=model_args)

    model.train_model(X)

    print(optimized_parameters)