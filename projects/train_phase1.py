# !pip install tensorflow numpy matplotlib nltk opencv-python fiftyone

import nltk
nltk.download('punkt')

import os
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence
import fiftyone.zoo as foz

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

MAX_SAMPLES = 5000
EPOCHS = 10

# Load COCO dataset
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    max_samples=MAX_SAMPLES
)

with open("src\captions_train2017.json", "r") as f:
    captions_data = json.load(f)["annotations"]

# Build image_id to captions map
id_to_caption = {}
for item in captions_data:
    image_id = item["image_id"]
    caption = item["caption"]
    if image_id not in id_to_caption:
        id_to_caption[image_id] = []
    id_to_caption[image_id].append(caption.lower())

image_paths = []
raw_captions = []

for sample in dataset:
    filename = os.path.basename(sample.filepath)
    try:
        image_id = int(os.path.splitext(filename)[0])
        if image_id in id_to_caption:
            for caption in id_to_caption[image_id]:
                image_paths.append(sample.filepath)
                raw_captions.append("<start> " + caption + " <end>")
    except Exception as e:
        print(f"Skipping {filename}: {e}")
        continue

print(f"Total samples: {len(image_paths)}")

# Tokenize captions
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(raw_captions)
sequences = tokenizer.texts_to_sequences(raw_captions)
padded_seqs = pad_sequences(sequences, padding="post")
max_len = padded_seqs.shape[1]
vocab_size = len(tokenizer.word_index) + 1

# Data generator
class ImageCaptionGenerator(Sequence):
    def __init__(self, image_paths, caption_seqs, batch_size=32, max_len=0):
        self.image_paths = image_paths
        self.caption_seqs = caption_seqs
        self.batch_size = batch_size
        self.max_len = max_len

    def __len__(self):
        return int(np.ceil(len(self.caption_seqs) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_seqs = self.caption_seqs[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        input_seqs = []
        targets = []

        for i, path in enumerate(batch_paths):
            img = Image.open(path).convert("RGB").resize((128, 128))
            img = np.array(img, dtype=np.float32) / 255.0
            images.append(img)

            seq = batch_seqs[i]
            input_seq = seq[:-1]
            target = seq[1:]

            input_seqs.append(input_seq)
            targets.append(target)

        images = np.array(images)
        input_seqs = pad_sequences(input_seqs, maxlen=self.max_len - 1)
        targets = pad_sequences(targets, maxlen=self.max_len - 1)

        return (images, input_seqs), targets

# CNN + LSTM model
model_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu")
])

image_input = tf.keras.Input(shape=(128, 128, 3))
x = model_cnn(image_input)

caption_input = tf.keras.Input(shape=(max_len - 1,))
y = tf.keras.layers.Embedding(vocab_size, 256)(caption_input)
y = tf.keras.layers.LSTM(256)(y)

combined = tf.keras.layers.Concatenate()([x, y])
z = tf.keras.layers.Dense(256, activation="relu")(combined)
z = tf.keras.layers.RepeatVector(max_len - 1)(z)
z = tf.keras.layers.LSTM(256, return_sequences=True)(z)
z = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation="softmax"))(z)

model = tf.keras.Model(inputs=[image_input, caption_input], outputs=z)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train
train_generator = ImageCaptionGenerator(
    image_paths=image_paths,
    caption_seqs=sequences,
    batch_size=16,
    max_len=max_len
)

model.fit(
    train_generator,
    epochs=EPOCHS,
)

# Save model, tokenizer, and max_len
model.save("src/caption_generator_model.h5")
with open("src/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("src/max_len.txt", "w") as f:
    f.write(str(max_len))

print("Model, tokenizer, and max_len saved.")