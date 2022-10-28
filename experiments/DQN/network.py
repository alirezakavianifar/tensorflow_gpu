import tensorflow as tf


class DeepQNetwork():
    def __init__(self, input_dims, n_actions, lr):
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.lr = lr
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(self.input_dims,)))
        model.add(tf.keras.layers.Dense(64), activation='relu')
        model.add(tf.keras.layers.Dense(64), activation='relu')
        model.add(tf.keras.layers.Dense(self.n_actions))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
                      loss=tf.keras.losses.MSE,
                      metrics=[tf.keras.metrics.Accuracy()])

        return model
