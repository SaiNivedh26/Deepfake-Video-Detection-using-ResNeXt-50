import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from huggingface.transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from huggingface.transformers.models import ResNext101
from huggingface.transformers.utils import download_model_file
from intel.openvino import OpenVINO

# Set up the OpenVINO environment
ov_env = OpenVINO()

# Download the ResNext101 model and tokenizer
model_file = download_model_file('hf_hub:timm/resnext101_32x16d.fb_swsl_ig1b_ft_in1k')
tokenizer_file = download_model_file('hf_hub:timm/resnext101_32x16d.fb_swsl_ig1b_ft_in1k')

# Load the ResNext101 model and tokenizer
model = ResNext101.from_pretrained(model_file)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_file)

# Load the FFHQ dataset
train_data = np.load('ffhq_train.npy')
test_data = np.load('ffhq_test.npy')

# Define the data generators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate the training and validation data
train_generator = train_datagen.flow_from_array(train_data, target_size=(224, 224), batch_size=32, class_mode='categorical')
val_generator = test_datagen.flow_from_array(test_data, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Define the image input shape
input_shape = (224, 224, 3)

# Define the model architecture
base_model = ResNet50(include_top=False, input_shape=input_shape)
for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(patience=5, monitor='val_loss')
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(train_generator, epochs=20, validation_data=val_generator, callbacks=[early_stopping, model_checkpoint])
