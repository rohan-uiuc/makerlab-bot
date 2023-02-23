from transformers import pipeline
import tensorflow as tf
import os

def train_model(model, tokenizer, data_path):
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=model.compute_loss, run_eagerly=True)

    print("Training model:{}".join(model.name))
    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        if os.path.splitext(file_path)[1] == ".txt":
            with open(file_path, 'r') as f:
                text = f.read()
                f.close()

            inputs = tokenizer(text, max_length=1024, padding=True, truncation=True, return_tensors='tf')

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            labels = input_ids[:, 1:]  # shift the labels to the right

            dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": input_ids, "attention_mask": attention_mask}, labels))
            dataset = dataset.batch(batch_size=8, drop_remainder=True)

            model.fit(dataset, epochs=1)

    print("Saving model")
    pipeline("text2text-generation", model=model, tokenizer=tokenizer, framework="tf").save_pretrained(model.name)
