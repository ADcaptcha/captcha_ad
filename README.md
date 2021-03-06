# captcha_ad
针对字符型图片验证码，使用tensorflow 2实现卷积神经网络，进行验证码识别。



##  目录结构
### 基本配置
| 序号 | 文件名称 | 说明 |
| ------ | ------ | ------ |
| 1 | `conf/` | 配置文件目录 |
| 2 | `sample/` | 数据集目录 |
| 3 | `save/` | 变量的保存与恢复 |
| 4 | `model/` | 模型文件目录 |
| 4 | `tb_logs/` | TensorBoard可视化 |

### 训练模型
| 序号 | 文件名称 | 说明 |
| ------ | ------ | ------ |
| 1 | gen_sample_by_captcha.py | 生成验证码的脚本 |
| 2 | verify_and_split_data.py | 验证数据集、拆分数据为训练集和测试集 |
| 3 | main.py | 训练模型 |

##  依赖

- tensorflow-gpu(tensorflow) >=2.3.0
- captcha==0.3

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

## 训练模型
创建好训练集和测试集之后，就可以开始训练模型了。  
训练的过程中会输出日志，日志展示当前的训练轮数、准确率和loss。
*此时的准确率是训练集图片的准确率，代表训练集的图片识别情况**  
例如：
```
第8000次训练 >>> 最高测试准确率为 0.94000
[训练集] 字符准确率为 0.98828 图片准确率为 0.96094 >>> loss 0.0186479837
[验证集] 字符准确率为 0.98500 图片准确率为 0.94000 >>> loss 0.0186479837
```