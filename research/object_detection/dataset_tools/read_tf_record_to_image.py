import tensorflow as tf

data_path = '/tmp/smart_fashion/body_parts_train.record-00000-of-00001'

with tf.Session() as sess:
    feature = {
        'image/to_string': tf.io.FixedLenFeature([], tf.string)
    }

    filename_queue = tf.train.string_input_producer([data_path], num_epochs=1)

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['image/to_string'], tf.float32)

    image = tf.reshape(image, [224, 224, 3])
    print("abc")
    image = sess.run(image)
    
    print(image)
    # images = tf.train.shuffle_batch
