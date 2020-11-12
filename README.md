# captcha_ad
针对字符型图片验证码，使用tensorflow 2实现卷积神经网络，进行验证码识别。

## 模型结构

| 序号 | 层级 |
| :------: | :------: |
| 输入 | input |
| 1 | 卷积层 + 池化层 + 降采样层 + swish  |
| 2 | 卷积层 + 池化层 + 降采样层 + swish  |
| 3 | 卷积层 + 池化层 + 降采样层 + swish  |
| 4 | 全连接 + 降采样层 + swish   |
| 5 | 全连接   |
| 输出 | output  |

## 执行顺序

1.python3 gen_sample_by_captcha.py

2.python3 verify_and_split_data.py

3.python3 main.py
