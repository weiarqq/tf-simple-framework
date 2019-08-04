import tensorflow as tf


# 加载模型参数
def init_checkpoints(modeling, model_path):
    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, model_path)
    tf.train.init_from_checkpoint(model_path, assignment_map)
    return initialized_variable_names