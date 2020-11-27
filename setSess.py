import os
import tensorflow as tf

from lib.setup import logging_config_setup, config_setup
from lib.model_utils import create_graph, load_weights, print_num_of_trainable_parameters
# from lib.train import train
from lib.train_sess import train

import argparse
import logging
from lib.utils import create_dir, check_path_exists
import json
import shap

def sess_params_setup():
    sess_parser = argparse.ArgumentParser()
    sess_parser.add_argument('--attention_len', type=int, default=16)
    sess_parser.add_argument('--batch_size', type=int, default=32)
    sess_parser.add_argument('--data_set', type=str, default='muse')
    sess_parser.add_argument('--decay', type=int, default=0)
    sess_parser.add_argument('--dropout', type=float, default=0.2)
    sess_parser.add_argument('--file_output', type=int, default=1)
    sess_parser.add_argument('--highway', type=int, default=0)
    sess_parser.add_argument('--horizon', type=int, default=3)
    sess_parser.add_argument('--init_weight', type=float, default=0.1)
    sess_parser.add_argument('--learning_rate', type=float, default=1e-5)
    sess_parser.add_argument('--max_gradient_norm', type=float, default=5.0)
    sess_parser.add_argument('--mode', type=str, default='train')
    sess_parser.add_argument('--model_dir', type=str, default='./models/model')
    sess_parser.add_argument('--mts', type=int, default=1)
    sess_parser.add_argument('--num_epochs', type=int, default=40)
    sess_parser.add_argument('--num_layers', type=int, default=3)
    sess_parser.add_argument('--num_units', type=int, default=338)

    para, unknown = sess_parser.parse_known_args()
    # para = parser.parse_args()
    para.mode = "validation"
    para.mode2 = "explain"
    para.attention_len = para.highway = 16
    para.horizon = 3
    para.data_set = "traffic"
    para.batch_size = 32
    para.learning_rate = 1e-3
    para.model_dir = "./models/traffic"
    para.num_epochs = 40
    para.num_units = 25

    if para.data_set == "muse" or para.data_set == "lpd5":
        para.mts = 0

    para.logging_level = logging.INFO

    if para.attention_len == -1:
        para.attention_len = para.max_len

    create_dir(para.model_dir)

    json_path = para.model_dir + '/parameters.json'
    json.dump(vars(para), open(json_path, 'w'), indent=4)
    return para


def main():
    para = sess_params_setup()
    logging_config_setup(para)

    graph, model, data_generator = create_graph(para)

    with tf.Session(config=config_setup(), graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        load_weights(para, sess, model)
        print_num_of_trainable_parameters()

        try:

            all_inputs, all_labels, all_outputs = train(para, sess, model, data_generator)

            return all_inputs, all_labels, all_outputs, model

            # all_inputs = all_inputs.permute(0,2,1)
            #
            # modelSize = 500
            # sampleSize = 500
            #
            # model_input = [tf.convert_to_tensor(x, dtype=tf.float32) for x in all_inputs[:modelSize]]
            # model_output = tf.convert_to_tensor(all_outputs[:modelSize], dtype=tf.float32)
            #
            # data = [x for x in all_inputs[:sampleSize]]
            # X = [x for x in all_inputs[:modelSize]]
            #
            # model2 = (model_input, model_output)
            # explainer = shap.DeepExplainer(model2, data, sess)
            # shap_values = explainer.shap_values(X)
            #
            # return shap_values

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        finally:
            print('Stop')


# if __name__ == '__main__':
#     os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#     main()
