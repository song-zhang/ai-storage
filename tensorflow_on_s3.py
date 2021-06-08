# -*- coding: utf-8 -*-
# vi:si:et:sw=4:sts=4:ts=4
# @Time    : 2021/4/8 9:01 PM
# @Author  : zhangsong

import os
import random
import tensorflow as tf
import boto3

# example:https://tensorflow.google.cn/tutorials/load_data/images?hl=zh_cn
# init s3 env for tensorflow s3 driver
os.environ['AWS_ACCESS_KEY_ID'] = "empty"
os.environ['AWS_SECRET_ACCESS_KEY'] = "empty"
os.environ['S3_ENDPOINT'] = "localhost:8333"
os.environ['S3_USE_HTTPS'] = "0"
os.environ['S3_VERIFY_SSL'] = "0"

# init s3 info for boto3 driver
aws_access_key_id = "empty"
aws_secret_access_key = "empty"
aws_endpoint_url = "http://localhost:8333"

bucket_name = "tensorflowbucket"
prefix = "flower_photos/"

AUTOTUNE = tf.data.experimental.AUTOTUNE


s3_client = boto3.client("s3", aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key,
                         endpoint_url=aws_endpoint_url)

# response structure of list_objects_v2():
# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.list_objects_v2
all_image_paths = list("s3://{}/{}".format(bucket_name, obj['Key']) for obj in
                       s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)['Contents'])
random.shuffle(all_image_paths)
label_names = sorted(prefix['Prefix'].rstrip('/').split('/')[-1] for prefix in
                     s3_client.list_objects_v2(Bucket=bucket_name, Delimiter='/', Prefix=prefix)['CommonPrefixes'])
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[path.split("/")[-2]] for path in all_image_paths]


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

BATCH_SIZE = 32

ds = image_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=len(all_image_paths)))
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)

mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
mobile_net.trainable = False


def change_range(image, label):
    return 2 * image - 1, label


keras_ds = ds.map(change_range)

model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_names), activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=["accuracy"])

steps_per_epoch = tf.math.ceil(len(all_image_paths) / BATCH_SIZE).numpy()

model.fit(ds, epochs=1, steps_per_epoch=3)
