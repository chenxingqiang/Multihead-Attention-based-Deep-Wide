#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import tensorflow as tf
from tensorflow import feature_column as fc

from bert_attention import *


def parse_arguments():
    """
    :return:
    """
    arguments_parse = argparse.ArgumentParser()
    arguments_parse.add_argument("--model_dir", type=str,
                                 help="Base directory for the model.")
    arguments_parse.add_argument("--train", type=str,default=''),
                                 help="Directory for storing  data")
    arguments_parse.add_argument("--test", type=str,default=''),
                                 help="Directory for storing  data")
        
    arguments_parse.add_argument("--hidden_units", type=str, default="128,64",
                                 help="Comma-separated list of number of units in each hidden NN layer ")
    arguments_parse.add_argument("--train_epoch", type=int, default=100, help="Number of training epochs.")
    arguments_parse.add_argument("--epoch_per_eval", type=int, default=1,
                                 help="The number of training epochs to run between evaluations.")
    arguments_parse.add_argument("--batch_size", type=int, default=256, help="Training batch size")
    arguments_parse.add_argument("--batch_norm", type=bool, default=True, help="batch_norm.")
    arguments_parse.add_argument("--shuffle_buffer_size", type=int, default=5000, help="dataset shuffle buffer size")
    arguments_parse.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    arguments_parse.add_argument("--dropout_rate", type=float, default=0.02, help="Drop out rate")
    arguments_parse.add_argument("--num_parallel_readers", type=int, default=16,
                                 help="number of parallel readers for training data")
    arguments_parse.add_argument("--save_checkpoints_steps", type=int, default=5000,
                                 help="Save checkpoints every this many steps")
    arguments_parse.add_argument('--model_type', type=str, default="wide_deep",
                                 help="choose model {'wide', 'deep', 'wide_deep'}")
    
    args, _ = arguments_parse.parse_known_args()
    
    print(args)

    return args

 args = parse_arguments()


def input_wide_deep_fc():
    """
    :return:
    """

    col_names = ["user_id", "item_id", "expotime", 
                 "sceneid", "user_seq_item_id",
                 "item_price", "item_cate_id", "item_type",
                 "item_mid_id","item_shopid", "item_position", 
                 "item_labelid", "item_state",
                 "item_statistic", "ctr_label"]
    print(col_names)
    
    
    # id feature to embedding
    user_id_hash = fc.categorical_column_with_hash_bucket("user_id", 10e8, dtype=tf.int64)
    item_id_hash = fc.categorical_column_with_hash_bucket("item_id", 5e5, dtype=tf.int64)
 
    # numeric values
    item_statistic_values = fc.numeric_column("item_statistic", shape=[20], dtype=tf.float32)

    # embeddings
    user_id_emb = fc.embedding_column(user_id_hash,32)
                                             
    item_id_share_emb = fc.shared_embedding_columns([item_id_hash, user_seq_item_id_hash], 32, combiner='sqrtn',
                                                  shared_embedding_collection_name="item_id_seq_shared_emb")
    
    wide_columns = [item__realtime_speakers_hash,
                    fc.categorical_column_with_hash_bucket("item_price", 100, dtype=tf.float32),
                    fc.categorical_column_with_hash_bucket("item_cate_id", 100, dtype=tf.string),
                    fc.categorical_column_with_hash_bucket("item_type", 100, dtype=tf.string),
                    fc.categorical_column_with_hash_bucket("item_shopid", 1e4, dtype=tf.string),
                    fc.categorical_column_with_hash_bucket("item_position", 2e2, dtype=tf.string),
                    fc.categorical_column_with_hash_bucket("item_labelid", 10e4, dtype=tf.string),
                    fc.categorical_column_with_hash_bucket("item_istop", 2, dtype=tf.string),
                    fc.categorical_column_with_hash_bucket("item_labelid", 14, dtype=tf.string),
                    fc.categorical_column_with_hash_bucket("item_state", 2, dtype=tf.string),
                    ]

    
    deep_columns = [user_id_emb,item_statistic_values,item_id_share_emb]
   
    return wide_columns, deep_columns

wide_columns, deep_columns = input_wide_deep_fc()

def input_parse_exmp(serial_exmp):
    """
    :param serial_exmp:
    :return:
    """
    click = fc.numeric_column("ctr_label", default_value=0, dtype=tf.int64)
    fea_columns = [click]
    global wide_columns
    global deep_columns
    
    fea_columns += wide_columns
    fea_columns += deep_columns

    feature_spec = tf.feature_column.make_parse_example_spec(fea_columns)
    
    other_feature_spec = {
        "expotime": tf.FixedLenFeature([], tf.string)}
    
    feature_spec.update(other_feature_spec)
    feats = tf.parse_single_example(serial_exmp, features=feature_spec)
    
    feats['num_item_ids'] = tf.count_nonzero(feats['user_seq_item_id'])
    feats['Max'] = tf.to_float(feats['item_statistic'][13])
    feats['min'] = tf.to_float(feats['item_statistic'][3])
    feats['hour'] = tf.strings.to_number(
        tf.strings.substr(feats['expotime'], 11, 2, name='hour', unit='UTF8_CHAR'))
    feats['minutes'] = tf.strings.to_number(
        tf.strings.substr(feats['expotime'], 14, 2, name='minutes', unit='UTF8_CHAR'))
    feats['item_statistic'] = tf.map_fn(tf.math.log1p, tf.cast(feats['item_statistic'], tf.float32))
    feats['item_ages'] = tf.map_fn(tf.math.reduce_max, feats['item_statistic'])

    feats['user_seqs'] = tf.expand_dims(feats['user_seq_item_id'].values,axis=0)
    feats['user_seqs'] = tf.expand_dims(feats['user_seqs'],axis=-1,name='user_seqs')
    feats['user_seqs'] = tf.cast(feats['user_seqs'],tf.float32)
  
    # user seq attention
    feats['seq_attion'] = attention_layer(from_tensor=feats['user_seqs'],
                                          to_tensor=feats['user_seqs'], 
                                          attention_mask=None,
                                          num_attention_heads=10,
                                          size_per_head=128,
                                          attention_probs_dropout_prob=0.2,
                                          do_return_2d_tensor=True,
                                          initializer_range=0.02,
                                          batch_size=args.batch_size,
                                          from_seq_length=20,
                                          to_seq_length=20
                                         )
    
    seq_attion = fc.numeric_column("seq_attion",dtype=tf.float32)
    bar_dead_ruler = fc.numeric_column("bar_dead_ruler",dtype=tf.float32)
    minutes = fc.numeric_column("minutes",dtype=tf.float32)

    deep_columns += [seq_attion]
    wide_columns += [bar_dead_ruler,minutes]
    
    click = feats.pop("ctr_label")

    return feats, tf.to_float(click)


def train_input_fn(filenames):
    print(filenames)
    files = tf.data.Dataset.list_files(filenames)
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset,
                                                              cycle_length=args.num_parallel_readers, 
                                                              sloppy=True))
    if args.shuffle_buffer_size > 0:
        dataset = dataset.shuffle(args.shuffle_buffer_size)
    dataset = dataset.map(input_parse_exmp, num_parallel_calls=16)
    dataset = dataset.repeat(args.train_epoch).batch(args.batch_size).prefetch(1)
    print(dataset.output_types)
    print(dataset.output_shapes)
    return dataset


def eval_input_fn(filenames):
    """
    :param filenames:
    :param arguments:
    :return:
    """
    print(filenames)
    files = tf.data.Dataset.list_files(filenames)
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset,
                                                              cycle_length=args.num_parallel_readers, 
                                                              sloppy=True))
    
    dataset = dataset.map(input_parse_exmp, num_parallel_calls=16)
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.batch(args.batch_size)
    # Return the read end of the pipeline.
    return dataset


def input_dataset_file():
    """
    :return:
    """
    train_data = args.train
    eval_data = args.test
    if isinstance(train_data, str) and os.path.isdir(train_data):
        train_files = [train_data + "/" + x for x in os.listdir(train_data)  
                       if "part" in x ] if os.path.isdir(
            train_data) else train_data
    else:
        train_files = train_data
    if isinstance(eval_data, str) and os.path.isdir(eval_data):
        eval_files = [eval_data + "/" + x for x in os.listdir(eval_data) 
                      if "part" in x ] if os.path.isdir(
            eval_data) else eval_data
    else:
        eval_files = eval_data
    return train_files, eval_files


def serving_input_fn():
    """
    Build the serving inputs.
    :return:
    """
    inputs = {"user_id": tf.placeholder(shape=[None, 1], dtype=tf.int64, name="user_id"),
              "item_id": tf.placeholder(shape=[None, 1], dtype=tf.int64, name="item_id"),
              "expotime": tf.placeholder(shape=[None, 1], dtype=tf.string, name="expotime"),
              "item_price": tf.placeholder(shape=[None, 1], dtype=tf.float32, name="item_price"),
              "item_cate_id": tf.placeholder(shape=[None, 1], dtype=tf.string, name="item_cate_id"),
              "item_type": tf.placeholder(shape=[None, 1], dtype=tf.string, name="item_type"),
              "item_mid_id": tf.placeholder(shape=[None, 1], dtype=tf.string, name="item_mid_id"),
              "item_position": tf.placeholder(shape=[None, 1], dtype=tf.string, name="item_position"),
              "item_shopid": tf.placeholder(shape=[None, 1], dtype=tf.string, name="item_shopid"),
              "item_labelid": tf.placeholder(shape=[None, 1], dtype=tf.string, name="item_labelid"),
              "item_state": tf.placeholder(shape=[None, 1], dtype=tf.string, name="item_state"),
              "user_seq_item_id": tf.placeholder(shape=[None, 1, 20], dtype=tf.int64, name="user_seq_item_id"),
              "item_statistic": tf.placeholder(shape=[None, 1, 20], dtype=tf.float32,
                                                      name="item_statistic")}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)


def build_estimator(run_config):
    """
    :param run_config:
    :return:
    """

    if args.hidden_units is None:
        hidden_units = [128, 64, 32]
    else:
        hidden_units = arguments.hidden_units.split(',')

    if args.model_type == "wide":
        model = tf.estimator.LinearClassifier(feature_columns=wide_columns,
                                              config=run_config)
    elif args.model_type == "deep":
        model = tf.estimator.DNNClassifier(
                                           feature_columns=deep_columns,
                                           hidden_units=hidden_units,
                                           batch_norm=args.batch_norm,
                                           config=run_config)
    else:

        model = tf.estimator.DNNLinearCombinedClassifier(
                                                         linear_feature_columns=wide_columns,
                                                         dnn_feature_columns=deep_columns,
                                                         dnn_hidden_units=hidden_units,
                                                         dnn_optimizer='Adam',
                                                         batch_norm=args.batch_norm,
                                                         config=run_config)
    return model


def main_fn():
    """
    :return:
    """

    cfg = tf.ConfigProto(log_device_placement=True)
    cfg.gpu_options.allow_growth = True

    run_cfg = tf.estimator.RunConfig().replace(model_dir=args.model_dir+"/checkpoint",
                                               session_config=cfg,
                                               keep_checkpoint_max=1,
                                               save_summary_steps=2000,
                                               save_checkpoints_steps=args.save_checkpoints_steps,
                                               log_step_count_steps=2000)

    model = build_estimator(run_cfg)
    train_files, eval_files = input_dataset_file()
    # train & evaluate
    model.train(input_fn=lambda: train_input_fn(train_files))
    results = model.evaluate(input_fn=lambda: eval_input_fn(eval_files), steps=5000)
    for key in sorted(results):
        print("%s : %s" % (key, results[key]))
    print("end of evaluate")
    print("exporting model ...")
    if arguments.serving_input =='handout':
        serving_input_receiver_fn = serving_input_fn
        model.export_savedmodel(args.model_dir+"/output_model", serving_input_receiver_fn)
        print("model saved")

    print("quit main")


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main_fn)

