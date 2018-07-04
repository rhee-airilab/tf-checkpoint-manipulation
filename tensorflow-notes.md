# Gems from Last AIRI-400 Lab Notes


## WEEK1_DAY4_Tensorboard_Summary_Note

`tf.summary.FileWriter(dir_path,graph)`

`tf.summary.scalar(a_tensor)`

`writer.add_summary()`


## WEEK1_DAY4_Tensorboard_Embedding_Note

`from tensorflow.contrib.tensorboard.plugins import projector`

`config = projector.ProjectorConfig()`

`embedding = config.embeddings.add()`

`embedding.tensor_name = 'dense/kernel'`

`embedding.metadata_path = 'metadata.tsv'`

`embedding.sprite.image_path = 'sprite.png'`

`embedding.sprite.single_image_dim.extend([28, 28])`

`projector.visualize_embeddings(writer, config)`


## WEEK1_DAY5_Tensorflow_Tips_Note

`config.gpu_options.allow_growth = True`

`CUDA_VISIBLE_DEVICES`

`tf.train.Saver()`

`tf.train.export_meta_graph()`

`tf.train.import_meta_graph()`

`inspect_checkpoint.py`

`python -mtensorflow.python.tools.inspect_checkpoint --file_name 'save/example-9361`

`tf.get_default_graph().finalize()`

`tf.reset_default_graph()`

`tf.check_numerics(tensor,messge)`

`python -m pdb myscript.py`

`from tensorflow.python import debug as tf_debug`

`sess = tf_debug.LocalCLIDebugWrapperSession(sess)`

## Week5-Day1-RNN-Basic-MNIST

`tf.contrib.rnn.BasicRNNCell()`

`tf.nn.dynamic_rnn()`

```
summary = tf.Summary(
            value=[
                tf.Summary.Value(
                    tag=tag,
                    simple_value=value)])
writer.add_summary(summary, step)
```

```
def log_images(self, tag, images, step):
    """Logs a list of images."""

    im_summaries = []
    for nr, img in enumerate(images):
        # Write the image to a string
        s = StringIO()
        plt.imsave(s, img, format='png')

        # Create an Image object
        img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                   height=img.shape[0],
                                   width=img.shape[1])
        # Create a Summary value
        im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                             image=img_sum))

    # Create and write Summary
    summary = tf.Summary(value=im_summaries)
    self.writer.add_summary(summary, step)
```

`https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514`


## Week5-Day2-RNN-static-rnn

`tf_batch_size = tf.shape(encoder_inputs)[0]`


## Week5-Day2-RNN-LSTM-GRU

```
from tensorflow.examples.tutorials.mnist.input_data \
    import read_data_sets
```

```
tvars_     = tf.trainable_variables()
grads_, _  = tf.clip_by_global_norm(
                tf.gradients(loss,tvars_),
                5.0)
optimize   = tf.train.AdamOptimizer(
                learning_rate=learning_rate) \
                .apply_gradients(zip(grads_, tvars_))
```


## Week5-Day2-RNN-Encoder-Decoder

```
seq_mask_ = tf.sequence_mask(
                decoder_seqlen,
                maxlen=decoder_max_seq_len,
                dtype=tf.float32)

seq_mask     = \
    tf.tile(
        tf.reshape(
            seq_mask_,
            [-1,decoder_max_seq_len,1]),
        [1,1,output_units])
```


## Week5_Day3_Motion_Generation

```
initial_state = tuple([
    tf.contrib.rnn.LSTMStateTuple(*tf.split(x, 2, axis=1))
    for x in tf.split(motion_state, num_layers, axis=1)])
```

