import numpy as np

from train.VAE.models.basic import VAEBasic


def _record_mse(metric, vae_image, org_image):
    """Record accuracy given predicted logits and ground-truth labels."""
    # softmax_probabilities = tf.keras.layers.Softmax()(logits)
    metric.update_state(org_image, vae_image)


def _record_loss(metric, loss):
    """Record accuracy given predicted logits and ground-truth labels."""
    # softmax_probabilities = tf.keras.layers.Softmax()(logits)
    metric.update_state(loss)


def create_model(conditioning_dim, flags):
    """Define model, and initialize classifiers."""
    model = VAEBasic(flags.height, flags.width, latent_dim=flags.latent_dim, conditioning_dim=conditioning_dim,
                     start_filters=flags.start_filters, name='VAE')

    return model


def _learning_rate_schedule(global_step_value, max_iters, initial_lr):
    """Calculates learning_rate with linear decay.

    Args:
    global_step_value: int, global step.
    max_iters: int, maximum iterations.
    initial_lr: float, initial learning rate.

    Returns:
    lr: float, learning rate.
    """

    if global_step_value <= max_iters//2:
        lr = initial_lr
    else:
        lr = initial_lr * np.exp(-0.001*(global_step_value - max_iters//2))
    return lr
