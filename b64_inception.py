import tensorflow as tf
import tensorflow_hub as hub

import base64
with open("./data/socks.jpg", "rb") as img_file:
    data = base64.urlsafe_b64encode(img_file.read())

jpeg_decode_fn = lambda x: tf.image.decode_jpeg(x, channels=3)
map_fn = lambda y: tf.cast(tf.map_fn(jpeg_decode_fn, tf.io.decode_base64(y),
                              dtype=tf.uint8), dtype=tf.float32)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(name='model_input', shape=[], dtype=tf.string))
b64_to_numpy = tf.keras.layers.Lambda(map_fn)
model.add(b64_to_numpy)

# b64 --> numpy vector
s1 = b64_to_numpy([data])
# model([data]) breaks

model_url = 'https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4'
inception = hub.KerasLayer(model_url, name='feature_vector')
print("Straight prediction with inception : ", inception(s1))

model.add(inception)

# save the model
model.save("./saved_model")

# model([data]) however does not work in eager execution.
# If you reload the model and execute, it works:
saved_model = tf.saved_model.load('./saved_model')
print("Prediction example with base64 layer added : ", saved_model([data]))
