import os

from algorithms.io.path_definition import get_project_dir


def get_tf_record_default_parameters():

    dir_train = f"{get_project_dir()}/data/train"

    train_directory = os.path.join(dir_train, 'img')
    val_directory = os.path.join(dir_train, 'img')
    test_directory = os.path.join(dir_train, 'img')
    output_directory = os.path.join(dir_train, 'tfrecord')

    train_csv_path = os.path.join(dir_train, 'image_train_list.csv')
    val_csv_path = os.path.join(dir_train, 'image_val_list.csv')
    test_csv_path = os.path.join(dir_train, 'image_test_list.csv')

    num_shards = 128
    image_size = 512

    default_path = {"TRAIN_DIRECTORY": train_directory,
                    "VAL_DIRECTORY": val_directory,
                    "TEST_DIRECTORY": test_directory,
                    "OUTPUT_DIRECTORY": output_directory,
                    "TRAIN_CSV_PATH": train_csv_path,
                    "VAL_CSV_PATH": val_csv_path,
                    "TEST_CSV_PATH": test_csv_path,
                    "NUM_SHARDS": num_shards,
                    "IMAGE_SIZE": image_size}

    return default_path


def get_train_default_parameters():

    dir_train = f"{get_project_dir()}/data/train"

    logdir = os.path.join(dir_train, 'fashion_v2')
    # train_file_pattern = os.path.join(dir_train, 'img', )
    seed = 0
    train_file_pattern = os.path.join(dir_train, "tfrecord", "train*")
    val_file_pattern = os.path.join(dir_train, "tfrecord", "val*")
    initial_lr = 0.01
    batch_size = 32
    max_iters = 100000
    use_augmentation = True
    clip_val = 1
    height = 256
    width = 256
    latent_dim = 256
    start_filters = 32

    default_path = {"LOGDIR": logdir,
                    "SEED": seed,
                    "TRAIN_FILE_PATTERN": train_file_pattern,
                    "VAL_FILE_PATTERN": val_file_pattern,
                    "INITIAL_LR": initial_lr,
                    "BATCH_SIZE": batch_size,
                    "MAX_ITERS": max_iters,
                    "USE_AUGMENTATION": use_augmentation,
                    "CLIP_VAL": clip_val,
                    "HEIGHT": height,
                    "WIDTH": width,
                    "LATENT_DIM": latent_dim,
                    "START_FILTERS": start_filters}

    return default_path