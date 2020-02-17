import tensorflow as tf
import tensorflow_hub as hub

# import model as Keraslayer
module_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
embed = hub.KerasLayer(module_url, name='embedding_output')

# create a new model
model = tf.keras.Sequential()

# add a layer for the input
model.add(tf.keras.layers.Input(name='model_input', shape=[], dtype=tf.string))

# add the loaded model
model.add(embed)

# save the model
model.save("./saved_model")