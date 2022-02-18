import argparse
import logging
import pickle
import time
from pathlib import Path
#
# Configure the path
ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / 'data'
MODELS_PATH = ROOT_PATH / 'models'
UTILS_PATH = ROOT_PATH / 'utils'

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import torch
from box import Box
from fastai.metrics import accuracy, error_rate
from fastai.text import (AWD_LSTM, TextClasDataBunch, TextLMDataBunch,
                         language_model_learner, text_classifier_learner)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('party prediction')


def fit_model(df, filename, fit_lm:True):
    
    LANGUAGEMODEL_FILE = MODELS_PATH / f'ft_enc_it_{filename}'

    df_trn, df_val = train_test_split(df,
                                      test_size=0.2,
                                      random_state=42)
    data_lm = TextLMDataBunch.from_df(train_df=df_trn,
                                      valid_df=df_val,
                                      path="",
                                      min_freq=1)
    learn = language_model_learner(data_lm,
                                   arch=AWD_LSTM,
                                   pretrained=True,
                                   drop_mult=0.3)
    if fit_lm:
        # Run one epoch with lower  layers
        logger.info('Fit frozen')
        learn.fit_one_cycle(5,
                            max_lr=1e-3, moms=(0.8, 0.7))
        learn.unfreeze()
        logger.info('Fit unfrozen')
        learn.fit_one_cycle(5,
                            max_lr=1e-3, moms=(0.8, 0.7))
        logger.info('Saving language model')
        learn.save_encoder(LANGUAGEMODEL_FILE)

    data_clas = TextClasDataBunch.from_df(path="",
                                          train_df=df_trn,
                                          valid_df=df_val,
                                          vocab=data_lm.train_ds.vocab,
                                          bs=64)
    data_clas.save(MODELS_PATH / f'databunch_{filename}')
    learn_clas = text_classifier_learner(data_clas,
                                         AWD_LSTM,
                                         drop_mult=0.5,
                                         metrics=[accuracy, error_rate])
    logger.info('Loading language model')
    learn_clas.load_encoder(LANGUAGEMODEL_FILE)
    lr = 1e-2
    lrm = 2.6
    lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm, lr])
    logger.info('Fit one cycle')
    learn_clas.fit_one_cycle(5, lrs)
    logger.info('Fit one cycle')
    learn_clas.freeze_to(-2)
    logger.info('Fit one cycle')
    learn_clas.fit_one_cycle(5, lrs)
    logger.info('Fit one cycle')
    learn_clas.freeze_to(-3)
    learn_clas.fit_one_cycle(2, lrs)
    learn_clas.freeze_to(-4)
    learn_clas.fit_one_cycle(2, lrs)
    return learn_clas


def obtain_party_classifier(df, text_column, label_column, substring_filename):
    df = df[[label_column, text_column]]
    model = fit_model(df, substring_filename, fit_lm=True)
    model.save(MODELS_PATH / f'classifier_{substring_filename}')
    model.predict("@berlusconi Come non condividere. Grande Silvio.")
    preds, targets = model.get_preds()
    predictions = np.argmax(preds, axis=1)
    logger.info(accuracy_score(targets, predictions))
    OUTPUT_FILE = MODELS_PATH / f'classifier_{substring_filename}_exported.pkl'
    model.export(OUTPUT_FILE)
    logger.info(f'Saving the model in file: {OUTPUT_FILE}')

