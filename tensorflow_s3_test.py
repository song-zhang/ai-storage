# -*- coding: utf-8 -*-
# vi:si:et:sw=4:sts=4:ts=4
# @Time    : 2021/4/12 9:03 PM
# @Author  : zhangsong


import os
import tensorflow as tf


from tensorflow.python.lib.io import file_io


os.environ['AWS_ACCESS_KEY_ID'] = "empty"
os.environ['AWS_SECRET_ACCESS_KEY'] = "empty"
os.environ['S3_ENDPOINT'] = "localhost:8333"
os.environ['S3_USE_HTTPS'] = "0"
os.environ['S3_VERIFY_SSL'] = "0"


#print(file_io.stat('s3://tensorflowbucket/flower_photos/tulips/9976515506_d496c5e72c.jpg'))
print(file_io.list_directory('s3://tensorflowbucket/flower_photos/'))

