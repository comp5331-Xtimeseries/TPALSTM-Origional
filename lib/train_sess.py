import time
import logging
import numpy as np
import tensorflow as tf

from lib.setup import config_setup
from lib.model_utils import save_model, create_valid_graph, load_weights


def train(para, sess, model, train_data_generator):
    valid_para, valid_graph, valid_model, valid_data_generator = \
        create_valid_graph(para)

    with tf.Session(config=config_setup(), graph=valid_graph) as valid_sess:
        valid_sess.run(tf.global_variables_initializer())

        # validation
        load_weights(valid_para, valid_sess, valid_model)
        valid_sess.run(valid_data_generator.iterator.initializer)
        valid_loss = 0.0
        valid_rse = 0.0
        valid_rae = 0.0
        count = 0
        n_samples = 0
        all_outputs, all_labels, all_inputs = [], [], []
        while True:
            try:
                [loss, outputs, labels, inputs] = valid_sess.run(fetches=[
                    valid_model.loss,
                    valid_model.all_rnn_outputs,
                    valid_model.labels,
                    valid_model.rnn_inputs
                ])
                if para.mts:
                    valid_rse += np.sum(
                        ((outputs - labels) * valid_data_generator.scale)
                        **2)
                    valid_rae += np.sum(
                        (abs(outputs - labels) * valid_data_generator.scale))
                    all_outputs.append(outputs)
                    all_labels.append(labels)
                    all_inputs.append(inputs)
                    n_samples += np.prod(outputs.shape)
                valid_loss += loss
                count += 1
            except tf.errors.OutOfRangeError:
                break
        if para.mts:
            all_outputs = np.concatenate(all_outputs)
            all_labels = np.concatenate(all_labels)
            all_inputs = np.concatenate(all_inputs)

    return all_inputs, all_labels, all_outputs
