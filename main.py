# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import time
import os
import json


def gen_captcha_text_image(img_path, img_name):
    """
    返回一个验证码的array形式和对应的字符串标签
    :return:tuple (str, numpy.array)
    """
    # 标签
    label = img_name.split("_")[0]
    # 文件
    img_file = os.path.join(img_path, img_name)
    captcha_image = Image.open(img_file)
    captcha_array = np.array(captcha_image)  # 向量化
    return label, captcha_array


def convert2gray(img):
    """
    图片转为灰度图，如果是3通道图则计算，单通道图则直接返回
    :param img:
    :return:
    """
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def text2vec(text):
    """
    转标签为oneHot编码
    :param text: str
    :return: numpy.array
    """
    text_len = len(text)
    if text_len > max_captcha:
        raise ValueError('验证码最长{}个字符'.format(max_captcha))

    vector = np.zeros(max_captcha * char_set_len)

    for i, ch in enumerate(text):
        idx = i * char_set_len + char_set.index(ch)
        vector[idx] = 1
    return vector


def get_batch(n, size=128):
    batch_x = np.zeros([size, image_height * image_width])  # 初始化
    batch_y = np.zeros([size, max_captcha * char_set_len])  # 初始化

    max_batch = int(len(train_images_list) / size)
    # print(max_batch)
    if max_batch - 1 < 0:
        raise TrainError("训练集图片数量需要大于每批次训练的图片数量")
    if n > max_batch - 1:
        n = n % max_batch
    s = n * size
    e = (n + 1) * size
    this_batch = train_images_list[s:e]
    # print("{}:{}".format(s, e))

    for i, img_name in enumerate(this_batch):
        label, image_array = gen_captcha_text_image(train_image_dir, img_name)
        image_array = convert2gray(image_array)  # 灰度化图片
        batch_x[i, :] = image_array.flatten() / 255.0  # flatten 转为一维
        batch_y[i, :] = text2vec(label)  # 生成 oneHot
    return batch_x, batch_y


def get_verify_batch(size=100):
    batch_x = np.zeros([size, image_height * image_width])  # 初始化
    batch_y = np.zeros([size, max_captcha * char_set_len])  # 初始化

    verify_images = []
    for i in range(size):
        verify_images.append(random.choice(verify_images_list))

    for i, img_name in enumerate(verify_images):
        label, image_array = gen_captcha_text_image(verify_image_dir, img_name)
        image_array = convert2gray(image_array)  # 灰度化图片
        batch_x[i, :] = image_array.flatten() / 255  # flatten 转为一维
        batch_y[i, :] = text2vec(label)  # 生成 oneHot
    return batch_x, batch_y


class AdModel(tf.keras.Model):
    def __init__(self,name=None):
        super().__init__(AdModel,name=name)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,  # 卷积层神经元（卷积核）数目
            kernel_size=[3, 3],  # 感受野大小
            padding='same',  # padding策略（vaild 或 same）
            activation=tf.nn.swish,
            name="conv1"

        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
        self.dropout1 = tf.keras.layers.Dropout(rate=keep_prob);
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.swish,
            name="conv2"
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
        self.dropout2 = tf.keras.layers.Dropout(rate=keep_prob);
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=[3, 3],
            padding='same',
            activation=tf.nn.swish,
            name="conv3"
        )
        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
        self.dropout3 = tf.keras.layers.Dropout(rate=keep_prob);


        self.flatten = tf.keras.layers.Reshape(target_shape=(8 * 13 * 128,))
        self.dense1 = tf.keras.layers.Dense(units=1024,
                                            activation=tf.nn.swish
                                            )
        self.dense2 = tf.keras.layers.Dense(units=max_captcha * char_set_len
                                            )

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        if training:
            x = self.dropout1(x,training=training)
        x = self.conv2(x)
        x = self.pool2(x)
        if training:
            x = self.dropout2(x, training=training)
        x = self.conv3(x)
        x = self.pool3(x)
        if training:
            x = self.dropout3(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        output= self.dense2(x)
        return output

    # 损失函数
    @tf.function
    def loss_func(self, features, y_true):
        with tf.GradientTape() as tape:
            y_pred = model(features, training=True)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true, name="loss"))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

        return loss

    # 评估指标(准确率)
    @tf.function
    def metric_func(self, y_pred,y_true):
        predict = tf.reshape(y_pred, [-1, max_captcha, char_set_len])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(y_true, [-1, max_captcha, char_set_len]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        acc_char_count = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        acc_image_count = tf.reduce_mean(tf.reduce_min(tf.cast(correct_pred, tf.float32), axis=1))
        return acc_char_count,acc_image_count

    #打印时间分割线
@tf.function
def printbar():
    today_ts = tf.timestamp()%(24*60*60)

    hour = tf.cast(today_ts//3600+8,tf.int32)%tf.constant(24)
    minite = tf.cast((today_ts%3600)//60,tf.int32)
    second = tf.cast(tf.floor(today_ts%60),tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}",m))==1:
            return(tf.strings.format("0{}",m))
        else:
            return(tf.strings.format("{}",m))

    timestring = tf.strings.join([timeformat(hour),timeformat(minite),
                timeformat(second)],separator = ":")
    tf.print("=========="*8+timestring)


if __name__ == '__main__':


    with open("conf/sample_config.json", "r") as f:
        sample_conf = json.load(f)

    train_image_dir = sample_conf["train_image_dir"]

    verify_image_dir = sample_conf["test_image_dir"]

    model_save_dir = sample_conf["model_save_dir"]

    acc_stop = sample_conf["acc_stop"]

    image_suffix = sample_conf['image_suffix']

    train_batch_size =  sample_conf['train_batch_size']

    test_batch_size = sample_conf['test_batch_size']

    char_set = sample_conf["char_set"]

    char_set = [str(i) for i in char_set]

    train_images_list = os.listdir(train_image_dir)
    # 打乱文件顺序
    random.seed(time.time())

    random.shuffle(train_images_list)

    verify_images_list = os.listdir(verify_image_dir)

    # 获得图片宽高和字符长度基本信息
    label, captcha_array = gen_captcha_text_image(train_image_dir, train_images_list[0])
    max_captcha = len(label)
    char_set_len = len(char_set)

    captcha_shape = captcha_array.shape
    captcha_shape_len = len(captcha_shape)
    if captcha_shape_len == 3:
        image_height, image_width, channel = captcha_shape
        channel = channel
    elif captcha_shape_len == 2:
        image_height, image_width = captcha_shape
    else:
        raise TrainError("图片转换为矩阵时出错，请检查图片格式")

    save_point =os.path.abspath(".") + '/save/model.ckpt'
    accuracy = 0.0
    keep_prob = 0.75
    is_save_model=True
    log_dir = 'tb_logs'
    model = AdModel("admodel")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    checkpoint = tf.train.Checkpoint(myAwesomeModel=model, myAwesomeOptimizer=optimizer)
    checkpoint.restore(tf.train.latest_checkpoint('./save'))

    summary_writer = tf.summary.create_file_writer(log_dir)  # 实例化记录器
    tf.summary.trace_on(graph=True, profiler=True)  # 开启Trace（可选）

    step = 1
    while True:
        batch_x, batch_y = get_batch(step, size=train_batch_size)
        batch_x = batch_x.astype(np.float32)
        batch_y = batch_y.astype(np.float32)
        X = tf.reshape(batch_x, shape=[-1, image_height, image_width, 1])

        loss = model.loss_func(X,batch_y)
        with summary_writer.as_default():  # 指定记录器
            tf.summary.scalar("loss", loss, step=step)  # 将当前损失函数的值写入记录器
        # 每100 step计算一次准确率
        if step % 10== 0:
            printbar()

            batch_x_test, batch_y_test  = get_batch(step, size=train_batch_size)
            batch_y_test = batch_y_test.astype(np.float32)
            batch_x_test = batch_x_test.astype(np.float32)
            batch_x_test = tf.reshape(batch_x_test, shape=[-1, image_height, image_width, 1])
            # model.training = False
            y_pred = model.predict(batch_x_test)
            train_acc_char_count,train_acc_image_count = model.metric_func(y_pred,batch_y_test)

            batch_x_verify, batch_y_verify = get_verify_batch(size=test_batch_size)
            batch_y_verify = batch_y_verify.astype(np.float32)
            batch_x_verify = batch_x_verify.astype(np.float32)
            batch_x_verify = tf.reshape(batch_x_verify, shape=[-1, image_height, image_width, 1])

            y_pred_verify= model.predict(batch_x_verify)
            vaild_acc_char_count,vaild_acc_image_count = model.metric_func(y_pred_verify, batch_y_verify)
            if(vaild_acc_image_count>accuracy):
                accuracy = vaild_acc_image_count
            print("第{}次训练 >>> 最高测试准确率为 {:.5f}".format(step, accuracy))
            print("[训练集] 字符准确率为 {:.5f} 图片准确率为 {:.5f} >>> loss {:.10f}".format(train_acc_char_count, train_acc_image_count, loss))

            print("[验证集] 字符准确率为 {:.5f} 图片准确率为 {:.5f} >>> loss {:.10f}".format(vaild_acc_char_count, vaild_acc_image_count, loss))
            with summary_writer.as_default():  # 指定记录器
                tf.summary.scalar("loss", loss, step=step)  # 将当前损失函数的值写入记录器
                tf.summary.scalar("Testaccuracy", train_acc_char_count, step=step,description="test")
                tf.summary.scalar("Trainaccuracy", vaild_acc_char_count, step=step,description="Train")

            if step % 1000 == 0:
                checkpoint.save(save_point)
                if (is_save_model):
                    # 保存Trace信息到文件（可选）
                    with summary_writer.as_default():
                        tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=log_dir)
                    is_save_model = False
                # break
                #
            # 如果准确率大于50%,保存模型,完成训练
            if accuracy > 0.999:
                    print("good nice")
                    tf.saved_model.save(model, './model/crack_capcha.h5')
                    break

        step += 1
        # 每100 step计算一次准确率



