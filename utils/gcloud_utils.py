import os

from tensorflow.python.lib.io import file_io


def copy_file_to_gcloud(local_file, job_dir, cloud_filename):
    with file_io.FileIO(local_file, mode='rb') as inf:
        with file_io.FileIO(os.path.join(job_dir, cloud_filename), mode='wb+') as of:
            of.write(inf.read())