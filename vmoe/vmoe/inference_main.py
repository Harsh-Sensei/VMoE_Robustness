# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main script."""
from typing import Sequence

from ml_collections import config_dict
from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags
import sys
import tensorflow as tf
import os
print(os.getcwd())
sys.path.append("./")
sys.path.append("../vision_transformer")
from vmoe.train import inference


flags.DEFINE_string('workdir', None, 'Directory to store logs and model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True)
flags.mark_flags_as_required(['config', 'workdir'])
FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], 'GPU')
  # Log JAX compilation steps.
  jax.config.update('jax_log_compiles', True)
  jax.config.update('jax_default_prng_impl', 'unsafe_rbg')
  # Log useful information to identify the process running in the logs.
  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())
  jax_xla_backend = ('None' if FLAGS.jax_xla_backend is None else
                     FLAGS.jax_xla_backend)
  logging.info('Using JAX XLA backend %s', jax_xla_backend)
  # Log the configuration passed to the main script.
  FLAGS.config.num_expert_partitions = 1
  FLAGS.config.dataset.test.data_dir = "./data_dir"
  FLAGS.config.dataset.test.manual_dir = "./manual_dir"
  FLAGS.config.initialization.prefix = "./vmoe/saved_checkpoints/ckpt_1" 
  tmp_dict = FLAGS.config.to_dict() #dataset.train
  del tmp_dict['dataset']['train']
  del tmp_dict['dataset']['test_real']
  del tmp_dict['dataset']['val']
  del tmp_dict['dataset']['imagenet_v2']

  FLAGS.config = config_dict.ConfigDict(tmp_dict)
  # del FLAGS.config.dataset.test_real
  # del FLAGS.config.dataset.val
  # del FLAGS.config.dataset.imagenet_v2
  
  logging.info('Config: %s', FLAGS.config)
  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
  platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                       FLAGS.workdir, 'workdir')
  # Select which mode to run.
  mode = FLAGS.config.get('mode', 'test')
  if mode == 'test':
    inference.evaluate(config=FLAGS.config, workdir=FLAGS.workdir)
  else:
    raise ValueError(f'Unknown mode: {FLAGS.config.mode!r}')


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
