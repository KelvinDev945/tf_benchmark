#!/usr/bin/env python3
"""测试TensorFlow BERT模型加载"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import TFBertForSequenceClassification
import tensorflow as tf

print('='*70)
print('测试TensorFlow BERT模型加载（h5格式）')
print('='*70)

model_name = 'google-bert/bert-base-uncased'
print(f'\n模型: {model_name}')
print('尝试加载...')

try:
    # 尝试加载h5格式的权重
    model = TFBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        from_pt=False,  # 不从PyTorch转换
        use_safetensors=False  # 不使用safetensors，使用h5
    )

    print('\n✓ 成功加载TensorFlow BERT模型!')
    print(f'  参数总数: {model.count_params():,}')
    print(f'  模型类型: {type(model).__name__}')

    # 测试推理
    print('\n测试推理...')
    import numpy as np

    dummy_input = {
        'input_ids': tf.constant([[101, 2023, 2003, 1037, 3231, 102]], dtype=tf.int32),
        'attention_mask': tf.constant([[1, 1, 1, 1, 1, 1]], dtype=tf.int32)
    }

    output = model(dummy_input)
    print(f'  ✓ 推理成功! 输出shape: {output.logits.shape}')

    print('\n'+'='*70)
    print('✓ TensorFlow BERT模型测试通过!')
    print('='*70)

except Exception as e:
    print(f'\n✗ 失败: {e}')
    import traceback
    traceback.print_exc()
