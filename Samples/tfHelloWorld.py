from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports

import numpy as np
import tensorflow as tf

# App

hello = tf.constant('Hello, TensorFlow!')
session = tf.Session()
print(session.run(hello))
session.close()