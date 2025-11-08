#!/usr/bin/env python3
"""
Test TensorFlow Engine fix using only TensorFlow/Keras (no transformers needed).

This test verifies that the TensorFlowEngine can load:
1. Models from string paths
2. Keras model objects directly
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engines.tensorflow_engine import TensorFlowEngine

print('=' * 70)
print('Testing TensorFlow Engine Type Check Fix')
print('=' * 70)

# Create a simple Keras model for testing
print('\n1. Creating a simple Keras model...')
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
print(f'   ✓ Model created: {type(model).__name__}')
print(f'   ✓ Has __call__: {hasattr(model, "__call__")}')
print(f'   ✓ Has predict: {hasattr(model, "predict")}')

# Test 1: Load Keras model object directly
print('\n2. Testing: Load Keras model object directly')
try:
    engine = TensorFlowEngine(config={'xla': False, 'mixed_precision': False})
    engine.load_model(model)
    print('   ✓ Successfully loaded Keras model object!')

    # Test inference
    dummy_input = np.random.randn(1, 10).astype(np.float32)
    output = engine.infer(dummy_input)
    print(f'   ✓ Inference successful! Output shape: {output.shape}')

    engine.cleanup()
except Exception as e:
    print(f'   ✗ FAILED: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Save and load model from path
print('\n3. Testing: Load model from SavedModel path')
try:
    # Save model
    model_path = '/tmp/test_keras_model'
    model.save(model_path)
    print(f'   ✓ Model saved to {model_path}')

    # Load from path
    engine = TensorFlowEngine(config={'xla': False, 'mixed_precision': False})
    engine.load_model(model_path)
    print('   ✓ Successfully loaded model from path!')

    # Test inference
    dummy_input = np.random.randn(1, 10).astype(np.float32)
    output = engine.infer(dummy_input)
    print(f'   ✓ Inference successful! Output shape: {output.shape}')

    engine.cleanup()

    # Cleanup
    import shutil
    shutil.rmtree(model_path)
except Exception as e:
    print(f'   ✗ FAILED: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test with h5 model file
print('\n4. Testing: Load model from .h5 file')
try:
    # Save as h5
    h5_path = '/tmp/test_model.h5'
    model.save(h5_path)
    print(f'   ✓ Model saved to {h5_path}')

    # Load from h5
    engine = TensorFlowEngine(config={'xla': False, 'mixed_precision': False})
    engine.load_model(h5_path)
    print('   ✓ Successfully loaded model from .h5 file!')

    # Test inference
    dummy_input = np.random.randn(1, 10).astype(np.float32)
    output = engine.infer(dummy_input)
    print(f'   ✓ Inference successful! Output shape: {output.shape}')

    engine.cleanup()

    # Cleanup
    os.remove(h5_path)
except Exception as e:
    print(f'   ✗ FAILED: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test with invalid input (should fail gracefully)
print('\n5. Testing: Invalid input type (should fail with clear error)')
try:
    engine = TensorFlowEngine(config={'xla': False, 'mixed_precision': False})
    engine.load_model(12345)  # Invalid type
    print('   ✗ Should have raised an error!')
    sys.exit(1)
except Exception as e:
    error_msg = str(e)
    if 'Invalid model_path type' in error_msg:
        print(f'   ✓ Correctly rejected invalid type with message:')
        print(f'     "{error_msg}"')
    else:
        print(f'   ✗ Wrong error message: {error_msg}')
        sys.exit(1)

print('\n' + '=' * 70)
print('✓ All TensorFlow Engine tests passed!')
print('=' * 70)
print('\nThe fix successfully supports:')
print('  - Keras model objects (using hasattr checks)')
print('  - SavedModel directories')
print('  - .h5 model files')
print('  - Proper error handling for invalid inputs')
print('\nThis means HuggingFace Transformers models should also work,')
print('as they have __call__ and predict methods like Keras models.')
print('=' * 70)
