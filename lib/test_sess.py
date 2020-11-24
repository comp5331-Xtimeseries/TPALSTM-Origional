import logging
import numpy as np
from tqdm import tqdm

def test(para, sess, model, data_generator):
    sess.run(data_generator.iterator.initializer)

    test_rse = 0.0
    test_rae = 0.0
    count = 0
    n_samples = 0
    all_outputs, all_labels, all_inputs = [], [],[]

    tp, fp, tn, fn = 0, 0, 0, 0
    while True:
        try:
            [outputs, labels, inputs] = sess.run(
                fetches=[
                    model.all_rnn_outputs,
                    model.labels,
                    model.rnn_inputs
                ]
            )

            test_rse += np.sum(
                ((outputs - labels) * data_generator.scale) ** 2
            )
            test_rae += np.sum(abs(outputs - labels) * data_generator.scale)
            all_outputs.append(outputs)
            all_labels.append(labels)
            all_inputs.append(inputs)
            count += 1
            n_samples += np.prod(outputs.shape)

            # print(outputs, labels, inputs)
        except Exception as e:
            # print("EXCEPTION AT TEST")
            # print(e)
            break

    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)
    all_inputs = np.concatenate(all_inputs)
    sigma_outputs = all_outputs.std(axis=0)
    sigma_labels = all_labels.std(axis=0)
    mean_outputs = all_outputs.mean(axis=0)
    mean_labels = all_labels.mean(axis=0)
    idx = sigma_labels != 0
    test_corr = (
        (all_outputs - mean_outputs) * (all_labels - mean_labels)
    ).mean(axis=0) / (sigma_outputs * sigma_labels)
    test_corr = test_corr[idx].mean()
    test_rse = (
        np.sqrt(test_rse / n_samples) / data_generator.rse
    )
    test_rae = ((test_rae) / data_generator.rae)
    # logging.info("test rse: %.5f, test_rae: %.5f, test corr: %.5f" % (test_rse, test_rae, test_corr))
    return all_inputs, all_labels, all_outputs
