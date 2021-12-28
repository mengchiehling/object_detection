import os
import re
from datetime import datetime
from typing import Dict, Tuple, List

import boto3

from algorithms.settings import logging, AWS_BUCKET, FEATURES, UPDATE_CONFIG, METADATA
from algorithms.io.metadata_definition import set_config
from algorithms.io.path_definition import get_project_dir

aws_bucket = AWS_BUCKET['VISUAL_SEARCH']


def set_aws_connection():

    """
    Setup connection to AWS s3

    Returns:
    AWS s3 connection
    """

    # %% AWS Credentials
    environment_config = set_config()
    aws_credential = environment_config['AWS']

    aws_access_key_id = aws_credential['aws_access_key_id']
    aws_secret_access_key = aws_credential['aws_secret_access_key']

    aws_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

    return aws_client


def transfer_aws_folder(aws_key_old: str, aws_key_new: str) -> int:

    """
    Transfer files between folders within AWS S3
    Move ALL the files in the folder specified by aws_key_old to a new directory.

    Args:
        aws_key_old: source folder to be transferred
        aws_key_new: destination folder
    Returns:
    number of files within the folder.
    """

    aws_client = set_aws_connection()
    resp = aws_client.list_objects_v2(Bucket=aws_bucket, Prefix=aws_key_old)
    last_index = len(resp['Contents'])
    for obj in resp['Contents']:
        f_source = obj['Key']
        f_dest = f"{aws_key_new}{f_source.replace(aws_key_old, '')}"
        aws_client.copy({"Bucket": aws_bucket, "Key": f_source}, aws_bucket, f_dest)

    return last_index + 1


def download_aws_folder(aws_key: str, local_folder: str):

    """
    Download all the files within a s3 folder to local hard disk

    Args:
        aws_key: source s3 folder to be downloaded
        local_folder: destination folder
    """

    if not os.path.isdir(local_folder):
        os.makedirs(local_folder)

    aws_client = set_aws_connection()
    resp = aws_client.list_objects_v2(Bucket=aws_bucket, Prefix=aws_key)

    for obj in resp['Contents']:
        key = obj['Key']
        file = key.split('/')[-1]
        logging.info(f": {key}")
        local_path = os.path.join(local_folder, file)
        aws_client.download_file(aws_bucket, key, local_path)


def upload_aws_file(local_file_name: str, aws_file_name: str):

    """
    Upload a file from local hard disk to AWS s3

    Args:
        local_file_name: filename in the local hard disk
        aws_file_name: filename (or key) in AWS s3
    """

    aws_client = set_aws_connection()
    aws_client.upload_file(local_file_name, aws_bucket, aws_file_name)


def download_aws_file(aws_file_name: str, local_file_name: str):
    """
    Download a file from AWS s3 to local hard disk

    Args:
        aws_file_name: filename (or key) in AWS s3
        local_file_name: filename in the local hard disk
    """

    aws_client = set_aws_connection()
    if not os.path.isfile(local_file_name):
        aws_client.download_file(aws_bucket, aws_file_name, local_file_name)


def get_all_folder_names(aws_key: str) -> List[str]:

    """
    Get the name of all the folders within aws_key

    Args:
        local_file_name: filename in the local hard disk
    Returns:
        A sorted list of directory
    """

    aws_client = set_aws_connection()
    dates = []
    resp = aws_client.list_objects_v2(Bucket=aws_bucket, Prefix=aws_key)
    for obj in resp['Contents']:
        obj_key = obj['Key']
        a = re.findall(r"\d{4}-\d{2}-\d{2}", obj_key)
        if len(a) != 0:
            dates.append(a[0])

    dates = list(set(dates))
    dates.sort()

    return dates


def get_latest_data_index(aws_key: str) -> str:

    """
    Get the latest data version in terms of date YYYY-MM-DD

    Args:
        aws_key:  AWS s3 key
    Returns:
        The full path of the latest folder in AWS
    """

    dates = get_all_folder_names(aws_key)
    latest_version = dates[-1]

    return latest_version


def remove_oldest_folder(aws_key: str, max_kept_folders: int):

    """
    Remove the oldest folder within aws_key to avoid data leakage

    Args:
        aws_key: AWS s3 key
        max_kept_folders: the maximum number of backup data we keep
    """

    dates = get_all_folder_names(aws_key)

    if len(dates) > max_kept_folders:
        oldest_version = dates[0]

        s3 = boto3.resource('s3')
        bucket = s3.Bucket(aws_bucket)
        f = bucket.objects.filter(Prefix=f"{aws_key}/{oldest_version}/").delete()


def build_file_path(fashion: str, mode: str, key: str) -> Tuple[Dict, Dict]:

    '''
    Building the paths of files in both AWS S3 and a local machine

    Args:
        fashion: SHOE or APPAREL
        key: AWS S3 key
    Returns:
        Path to local drive and aws in dictionaries
    '''

    dir_feature = os.path.join(get_project_dir(), 'data', 'app', 'features')
    dir_metadata = os.path.join(get_project_dir(), 'data', 'app', 'metadata')

    if mode == 'recompute':
        now = datetime.now().strftime("%Y-%m-%d")
        latest_version = now
        aws_folder = os.path.join(key, latest_version)
    elif mode == 'update':
        latest_version = UPDATE_CONFIG['DATABASE_VERSION']
        if latest_version == "latest":
            latest_version = get_latest_data_index(key)
        aws_folder = os.path.join(key, latest_version)
    elif mode == 'load':
        latest_version = None
        aws_folder = os.path.join(key, 'features')
    else:
        raise RuntimeError(f"mode {mode} has no corresponding usage")

    # use python comprehensor to shorten the code
    aws_path_dict = {v: os.path.join(aws_folder, f'{fashion}-{v}.csv') for _, v in FEATURES.items()}
    aws_path_dict.update({v: os.path.join(aws_folder, f'{fashion}-{v}.csv') for _, v in METADATA.items()})

    if latest_version is None:
        local_path_dict = {v: os.path.join(dir_feature, f'{fashion}-{v}.csv') for _, v in FEATURES.items()}
        local_path_dict.update({v: os.path.join(dir_metadata, f'{fashion}-{v}.csv') for _, v in METADATA.items()})
    else:
        local_path_dict = {v: os.path.join(dir_feature, f'{fashion}-{v}-{latest_version}.csv')
                           for _, v in FEATURES.items()}
        local_path_dict.update({v: os.path.join(dir_metadata, f'{fashion}-{v}-{latest_version}.csv')
                                for _, v in METADATA.items()})

    return aws_path_dict, local_path_dict
