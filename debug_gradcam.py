
import os
import numpy as np
import tensorflow as tf
from PIL import Image

def apply_clahe(image_uint8, clip_limit=1.5, tile_grid_size=(8, 8)):
    import cv2
    lab = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

print("Loading model...")
model = tf.keras.models.load_model('Frozen92.16.keras', compile=False, custom_objects={'Custom>mish': tf.keras.activations.mish, 'mish': tf.keras.activations.mish})
SONO_IMG_SIZE = 300

# Replicate what sono_last_conv_layer does
sono_last_conv_layer = None
for layer in reversed(model.layers):
    out_shape = layer.output.shape
    if isinstance(out_shape, list):
        out_shape = out_shape[0].shape
    if len(out_shape) == 4 and out_shape[1] is not None and out_shape[1] > 1:
        sono_last_conv_layer = layer.name
        break

print("Using last conv layer:", sono_last_conv_layer)

grad_model = tf.keras.models.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(sono_last_conv_layer).output, model.layers[-2].output]
)
dense_weights, dense_biases = model.layers[-1].get_weights()

def run_gradcam(arr):
    img_batch = tf.cast(np.expand_dims(arr, 0), tf.float32)
    preds = model.predict(img_batch, verbose=0)
    pred_class = int(np.argmax(preds[0]))
    
    with tf.GradientTape() as tape:
        tape.watch(img_batch)
        conv_outputs, dense_features = grad_model(img_batch)
        logits = tf.matmul(dense_features, dense_weights) + dense_biases
        class_score = logits[:, pred_class]
    
    grads = tape.gradient(class_score, conv_outputs)[0]
    if grads is None:
        print("GRADS ARE NONE!")
        return None
        
    activations = conv_outputs[0]
    grads_power_2 = grads ** 2
    grads_power_3 = grads_power_2 * grads
    sum_activations = tf.reduce_sum(activations, axis=(0, 1))
    eps = 1e-7
    denom = 2.0 * grads_power_2 + sum_activations * grads_power_3 + eps
    aij = grads_power_2 / denom
    aij = tf.where(grads != 0.0, aij, 0.0)
    weights = tf.maximum(grads, 0.0) * aij
    weights = tf.reduce_sum(weights, axis=(0, 1))
    cam_raw = tf.reduce_sum(weights * activations, axis=-1).numpy()
    
    print(f"Pred class: {pred_class}, score: {class_score.numpy()[0]:.4f}")
    print("max grads:", np.max(grads), "min grads:", np.min(grads))
    print("max weights:", np.max(weights), "min weights:", np.min(weights))
    print("max cam_raw before max(0):", np.max(cam_raw))
    
    cam_raw = np.maximum(cam_raw, 0)
    print("max cam_raw after max(0):", np.max(cam_raw))
    
    return cam_raw

dummy_noise = np.random.randint(0, 256, (SONO_IMG_SIZE, SONO_IMG_SIZE, 3), dtype=np.uint8)
arr_noise = apply_clahe(dummy_noise).astype(np.float32)
print("Running Grad-CAM...")
run_gradcam(arr_noise)
