#!/usr/bin/env python3
"""
多线程Benchmark Worker脚本

在导入TensorFlow之前设置线程配置
"""

import os
import sys
import time
import json
import numpy as np

# 从命令行参数获取线程配置
intra_threads = int(sys.argv[1])
inter_threads = int(sys.argv[2])
model_type = sys.argv[3]
num_runs = int(sys.argv[4])
batch_size = int(sys.argv[5])

# 设置环境变量（必须在导入TensorFlow之前）
os.environ['TF_NUM_INTRAOP_THREADS'] = str(intra_threads) if intra_threads > 0 else ''
os.environ['TF_NUM_INTEROP_THREADS'] = str(inter_threads) if inter_threads > 0 else ''

# 现在导入TensorFlow
import tensorflow as tf

# 验证线程设置
if intra_threads > 0:
    tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
if inter_threads > 0:
    tf.config.threading.set_inter_op_parallelism_threads(inter_threads)

def create_bert_base_model(seq_length=128, vocab_size=10000):
    """创建BERT-Base模型"""
    hidden_size = 768
    num_hidden_layers = 12
    num_attention_heads = 12
    intermediate_size = hidden_size * 4

    input_ids = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32, name='input_ids')

    embeddings = tf.keras.layers.Embedding(
        vocab_size, hidden_size, name='embedding'
    )(input_ids)

    position_embeddings = tf.keras.layers.Embedding(
        seq_length, hidden_size, name='position_embedding'
    )(tf.range(seq_length))

    x = embeddings + position_embeddings
    x = tf.keras.layers.LayerNormalization(epsilon=1e-12)(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    for i in range(num_hidden_layers):
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_attention_heads,
            key_dim=hidden_size // num_attention_heads,
            name=f'attention_{i}'
        )(x, x)

        attention_output = tf.keras.layers.Dropout(0.1)(attention_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-12)(x + attention_output)

        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(intermediate_size, activation='relu'),
            tf.keras.layers.Dense(hidden_size)
        ], name=f'ffn_{i}')

        ffn_output = ffn(x)
        ffn_output = tf.keras.layers.Dropout(0.1)(ffn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-12)(x + ffn_output)

    pooled_output = tf.keras.layers.Lambda(lambda x: x[:, 0])(x)
    pooled_output = tf.keras.layers.Dense(
        hidden_size, activation='tanh', name='pooler'
    )(pooled_output)

    output = tf.keras.layers.Dense(2, activation='softmax', name='classifier')(pooled_output)

    model = tf.keras.Model(inputs=input_ids, outputs=output, name='bert_base_model')
    return model

def create_mobilenet_model(input_shape=(224, 224, 3), num_classes=1000):
    """创建MobileNetV2模型"""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=None
    )

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name='mobilenet_v2')

    return model

def benchmark():
    """执行benchmark测试"""
    # 创建模型
    if model_type == "bert":
        model = create_bert_base_model()
        X_test = np.random.randint(0, 10000, size=(200, 128), dtype=np.int32)
        y_test = np.random.randint(0, 2, 200)
    else:  # mobilenet
        model = create_mobilenet_model()
        X_test = np.random.randn(200, 224, 224, 3).astype(np.float32)
        y_test = np.random.randint(0, 1000, 200)

    # 编译模型
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 热身
    num_warmup = 5
    for _ in range(num_warmup):
        _ = model.predict(X_test[:batch_size], verbose=0)

    # 性能测试
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model.predict(X_test[:batch_size], verbose=0)
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

    # 准确率测试
    loss, accuracy = model.evaluate(X_test[:100], y_test[:100], verbose=0)

    # 统计
    latencies = np.array(latencies)
    results = {
        "intra_threads": intra_threads,
        "inter_threads": inter_threads,
        "batch_size": batch_size,
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "std_ms": float(np.std(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "throughput_samples_per_sec": float(batch_size * 1000.0 / np.mean(latencies)),
        "accuracy": float(accuracy)
    }

    # 输出JSON结果到stdout
    print(json.dumps(results))

if __name__ == "__main__":
    benchmark()
