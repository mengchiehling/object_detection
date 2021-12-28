from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1.keras.losses import mse
from tensorflow.keras.metrics import MeanSquaredError, Mean

from algorithms.io.path_definition import get_project_dir
import train.VAE.utils_tf_data_parsing as gld
from train.VAE.utils_train import create_model, _learning_rate_schedule
from train.VAE.utils_default_parameters import get_train_default_parameters

dir_train = f"{get_project_dir()}/data/train"
default_parameters = get_train_default_parameters()

FLAGS = flags.FLAGS

flags.DEFINE_boolean('debug', False, 'Debug mode.')
flags.DEFINE_string('logdir', default_parameters['LOGDIR'], 'WithTensorBoard logdir.')
flags.DEFINE_string('train_file_pattern', default_parameters['TRAIN_FILE_PATTERN'],
                    'File pattern of train dataset files.')
flags.DEFINE_string('val_file_pattern', default_parameters['VAL_FILE_PATTERN'],
                    'File pattern of validation dataset files.')
flags.DEFINE_integer('seed', default_parameters['SEED'], 'Seed to train dataset.')
flags.DEFINE_float('initial_lr', default_parameters['INITIAL_LR'], 'Initial learning rate.')
flags.DEFINE_integer('batch_size', default_parameters['BATCH_SIZE'], 'Global batch size.')
flags.DEFINE_integer('max_iters', default_parameters['MAX_ITERS'], 'Maximum iterations.')
flags.DEFINE_boolean('use_augmentation', default_parameters['USE_AUGMENTATION'],
                     'Whether to use ImageNet style augmentation.')
flags.DEFINE_float('clip_val', default_parameters['CLIP_VAL'], 'gradient clip')
flags.DEFINE_integer('height', default_parameters['HEIGHT'], 'image height')
flags.DEFINE_integer('width', default_parameters['WIDTH'], 'image width')
flags.DEFINE_integer('latent_dim', default_parameters['LATENT_DIM'], 'latent space dimensionality')
flags.DEFINE_integer('start_filters', default_parameters['START_FILTERS'], 'start filters')


def main(argv):

    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # -------------------------------------------------------------
    # Log flags used.
    logging.info('Running train script with\n')
    logging.info('logdir= %s', FLAGS.logdir)
    logging.info('initial_lr= %f', FLAGS.initial_lr)
    # ------------------------------------------------------------
    # Create the strategy.
    strategy = tf.distribute.MirroredStrategy()
    logging.info('Number of devices: %d', strategy.num_replicas_in_sync)
    if FLAGS.debug:
        print('Number of devices:', strategy.num_replicas_in_sync)

    max_iters = FLAGS.max_iters
    global_batch_size = FLAGS.batch_size
    num_eval = 100
    report_interval = 100
    eval_interval = 100
    save_interval = 100
    initial_lr = FLAGS.initial_lr

    clip_val = tf.constant(FLAGS.clip_val)

    if FLAGS.debug:
        global_batch_size = 16
        max_iters = 4
        num_eval = 1
        save_interval = 1
        report_interval = 1

    train_dataset = gld.create_dataset(file_pattern=FLAGS.train_file_pattern,
                                       batch_size=global_batch_size,
                                       image_height=FLAGS.height,
                                       image_width=FLAGS.width,
                                       augmentation=FLAGS.use_augmentation,
                                       seed=FLAGS.seed)

    val_dataset = gld.create_dataset(file_pattern=FLAGS.val_file_pattern,
                                     batch_size=global_batch_size,
                                     image_height=FLAGS.height,
                                     image_width=FLAGS.width,
                                     augmentation=False,
                                     seed=FLAGS.seed)

    train_iterator = strategy.make_dataset_iterator(train_dataset)
    validation_iterator = strategy.make_dataset_iterator(val_dataset)

    train_iterator.initialize()
    validation_iterator.initialize()

    # Create a checkpoint directory to store the checkpoints.
    checkpoint_prefix = os.path.join(FLAGS.logdir, f'fashion_{FLAGS.latent_dim}_tf2-ckpt')

    with strategy.scope():
        # Compute loss.
        # Set reduction to `none` so we can do the reduction afterwards and divide
        # by global batch size.

        def compute_loss(x, x_decoded_mean, z_mean, z_log_sigma):
            mse_loss = K.mean(mse(x, x_decoded_mean), axis=(1, 2)) * FLAGS.height * FLAGS.width
            kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
            return mse_loss + kl_loss

        # Set up metrics.
        val_total_loss = Mean(name='val_loss')
        val_pixel_mse_loss = MeanSquaredError(name='val_pixel_mse')
        train_total_loss = Mean(name='train_loss')
        train_pixel_mse_loss = MeanSquaredError(name='train_pixel_mse')

        model = create_model(0, FLAGS)
        logging.info('Model, datasets loaded.\n')
        optimizer = tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)

        # Setup summary writer.
        summary_writer = tf.summary.create_file_writer(
            os.path.join(FLAGS.logdir, 'train_logs'), flush_millis=10000)

        # Setup checkpoint directory.
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(
            checkpoint, checkpoint_prefix, max_to_keep=3)

        # Train step to run on one GPU.
        def train_step(inputs):
            """Train one batch."""
            # try:
            #     images, conditions = inputs
            # except ValueError:
            images = inputs
            conditions = None
            # Temporary workaround to avoid some corrupted labels.

            global_step = optimizer.iterations
            tf.summary.scalar(
                'image_range/max', tf.reduce_max(images), step=global_step)
            tf.summary.scalar(
                'image_range/min', tf.reduce_min(images), step=global_step)

            def _backprop_loss(tape, loss, weights):
                """Backpropogate losses using clipped gradients.

                Args:
                  tape: gradient tape.
                  loss: scalar Tensor, loss value.
                  weights: keras model weights.
                """
                gradients = tape.gradient(loss, weights)
                clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=clip_val)
                optimizer.apply_gradients(zip(clipped, weights))

            # Record gradients and loss through backbone.
            with tf.GradientTape() as desc_tape:

                vae_images, z_mean, z_log_sigma = model(images, conditions, training=True)

                train_loss = compute_loss(images, vae_images, z_mean, z_log_sigma)

                # Backprop only through backbone weights.
            _backprop_loss(desc_tape, train_loss, model.trainable_weights)

            # Record descriptor train accuracy.
            train_pixel_mse_loss.update_state(images, vae_images) # update train loss on the pixel level
            train_total_loss.update_state(train_loss) # update train loss on both image + regularization

            return train_total_loss.result(), train_pixel_mse_loss.result()

        def validation_step(inputs):
            """Validate one batch."""
            images = inputs
            conditions = None

            # Get descriptor predictions.
            vae_images, z_mean, z_log_sigma = model(images, conditions, training=True)

            val_total_loss.update_state(compute_loss(images, vae_images, z_mean, z_log_sigma))
            val_pixel_mse_loss.update_state(images, vae_images)

            return val_total_loss.result(), val_pixel_mse_loss.result()

        @tf.function
        def distributed_train_step(dataset_inputs):
            """Get the actual losses."""
            # Each (desc, attn) is a list of 3 losses - crossentropy, reg, total.
            # desc_per_replica_loss = (strategy.run(train_step, args=(dataset_inputs,)))

            per_replica_total_loss, per_replica_pixel_mse_loss = strategy.run(train_step, args=(dataset_inputs,))

            return per_replica_total_loss, per_replica_pixel_mse_loss

        @tf.function
        def distributed_validation_step(dataset_inputs):
            return strategy.run(validation_step, args=(dataset_inputs,))

        with summary_writer.as_default():
            with tf.summary.record_if(
                    tf.math.equal(0, optimizer.iterations % report_interval)):

                global_step_value = optimizer.iterations.numpy()
                while global_step_value < max_iters:

                    # input_batch : images(b, h, w, c), labels(b,).
                    try:
                        input_batch = train_iterator.get_next()
                    except tf.errors.OutOfRangeError:
                        # Break if we run out of data in the dataset.
                        logging.info('Stopping train at global step %d, no more data',
                                     global_step_value)
                        break

                    # Set learning rate for optimizer to use.
                    global_step = optimizer.iterations
                    global_step_value = global_step.numpy()

                    learning_rate = _learning_rate_schedule(global_step_value, max_iters, initial_lr)
                    optimizer.learning_rate = learning_rate
                    tf.summary.scalar('learning_rate', optimizer.learning_rate, step=global_step)

                    # Run the train step over num_gpu gpus.
                    total_train_loss_result, pixel_train_mse_loss_result = distributed_train_step(input_batch)

                    # Log losses and accuracies to tensorboard.
                    tf.summary.scalar('loss/total', total_train_loss_result, step=global_step)
                    tf.summary.scalar('loss/pixel mse loss', pixel_train_mse_loss_result, step=global_step)
                    # Print to console if running locally.
                    if FLAGS.debug:
                        if global_step_value % report_interval == 0:
                            print(global_step.numpy())
                            print('total_loss:', total_train_loss_result.numpy())

                    # Validate once in {eval_interval*n, n \in N} steps.
                    if (global_step_value % eval_interval == 0):
                        for i in range(num_eval):
                            try:
                                validation_batch = validation_iterator.get_next()
                                total_val_loss_result, pixel_val_mse_loss_result = (
                                    distributed_validation_step(validation_batch))
                            except tf.errors.OutOfRangeError:
                                logging.info('Stopping eval at batch %d, no more data', i)
                                break
                        # Log validation results to tensorboard.
                        tf.summary.scalar('val/total loss', total_val_loss_result, step=global_step)
                        tf.summary.scalar('val/pixel mse loss', pixel_val_mse_loss_result, step=global_step)

                        logging.info('\nValidation(%f)\n', global_step_value)
                        logging.info(': train total loss: %f', total_train_loss_result.numpy())
                        logging.info(': train pixel mse loss: %f', pixel_train_mse_loss_result.numpy())
                        logging.info(': val total loss: %f', total_val_loss_result.numpy())
                        logging.info(': val pixel mse loss %f', pixel_val_mse_loss_result.numpy())
                        # Print to console.
                        if FLAGS.debug:
                            print('Val: total loss:', total_val_loss_result.numpy())
                            print('   : pixel loss:', pixel_val_mse_loss_result.numpy())

                    # Save checkpoint once (each save_interval*n, n \in N) steps.
                    if global_step_value % save_interval == 0:
                        save_path = manager.save()
                        logging.info(f'Saved({global_step_value}) at %s', save_path)

                        file_path = f'{FLAGS.logdir}/fashion_{FLAGS.latent_dim}_weights'
                        model.save_weights(file_path, save_format='tf')
                        logging.info(f'Saved weights({global_step_value}) at %s', file_path)
                    # Reset metrics for next step.
                    val_total_loss.reset_states()
                    val_pixel_mse_loss.reset_states()
                    train_pixel_mse_loss.reset_states()
                    train_total_loss.reset_states()

                    if global_step.numpy() > max_iters:
                        break

        logging.info('Finished train for %d steps.', max_iters)


if __name__ == '__main__':

    app.run(main)