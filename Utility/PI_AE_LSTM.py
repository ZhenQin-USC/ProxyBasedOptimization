import sys
sys.path.append(r'D:\Users\qinzh\Google Drive USC\MATLAB Local\Proxy Opt')
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class PI_AELSTM(tf.keras.Model):
    def __init__(self, nz, nc, NX=21, NY=15, NZ=1):
        super(PI_AELSTM, self).__init__()
        self.nz = nz
        self.nc = nc
        self.NX = NX
        self.NY = NY
        self.NZ = NZ
        self.paddings = tf.constant([[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]])
        self.LX = tf.constant(6300)
        self.LY = tf.constant(4500)
        self.dx = tf.cast(self.LX / self.NX, dtype=tf.float32)
        self.dy = tf.cast(self.LY / self.NY, dtype=tf.float32)
        self.dz = tf.constant(100)
        self.kx = tf.constant(50)
        self.ky = tf.constant(20)
        self.k = tf.cast(np.sqrt(self.kx * self.ky), dtype=tf.float32)
        self.phi = tf.cast(0.2, dtype=tf.float32)
        self.cf = tf.cast(1.25 * 1e-5, dtype=tf.float32)
        self.co = tf.cast(1.75 * 1e-5, dtype=tf.float32)
        self.ct = tf.cast(self.cf + self.co, dtype=tf.float32)
        self.h = tf.constant(self.dz)
        self.mu = tf.constant(5)
        self.B = tf.cast(1.25, dtype=tf.float32)
        self.rw = tf.cast(0.25, dtype=tf.float32)
        re = 0.28 * (np.sqrt(self.dx ** 2 * self.ky + self.dy ** 2 * self.kx)) / (np.sqrt(self.kx) + np.sqrt(self.ky))
        self.re = tf.cast(re,dtype=tf.float32)
        self.S = tf.constant(0)

        self.Jw = 7.08 * 1e-3 * (self.k * self.h / (self.mu * self.B)) * (1. / (np.log(self.re / self.rw) + self.S))
        self.lam_x = self.kx / self.mu
        self.lam_y = self.ky / self.mu
        self.a = 158 * self.phi * self.ct
        self.gamma = (887.17 * self.B / self.dz)
        # self.model = self.ae_lstm()
        # return self.model

    def encoder(self):
        input_img = tf.keras.Input(batch_shape=(None, self.NX, self.NY, self.NZ), name='InitialMap')
        x = layers.Conv2D(filters=4, kernel_size=(6, 5), strides=(1, 1), activation='relu')(input_img)
        x = layers.Conv2D(filters=6, kernel_size=(5, 4), strides=(1, 1), activation='relu')(x)
        x = layers.Conv2D(filters=8, kernel_size=(4, 3), strides=(1, 1), activation='relu')(x)
        x = layers.Conv2D(filters=10, kernel_size=(3, 2), strides=(1, 1), activation='relu')(x)
        x = layers.Conv2D(filters=12, kernel_size=3, strides=(1, 1), activation='relu')(x)
        x = layers.Flatten()(x)
        zm = layers.Dense(self.nz)(x)  # encoded image
        model = tf.keras.Model(inputs=input_img, outputs=zm)
        return model

    def decoder(self):
        zm = tf.keras.Input(batch_shape=(None, self.nz))
        x = layers.Dense(180)(zm)
        x = layers.Reshape(target_shape=(5, 3, 12))(x)
        x = layers.Conv2DTranspose(filters=12, kernel_size=3, strides=(1, 1), activation='relu')(x)
        x = layers.Conv2DTranspose(filters=10, kernel_size=(3, 2), strides=(1, 1), activation='relu')(x)
        x = layers.Conv2DTranspose(filters=8, kernel_size=(4, 3), strides=(1, 1), activation='relu')(x)
        x = layers.Conv2DTranspose(filters=6, kernel_size=(5, 4), strides=(1, 1), activation='relu')(x)
        x = layers.Conv2DTranspose(filters=4, kernel_size=(6, 5), strides=(1, 1), activation='relu')(x)
        decoded_img = layers.Conv2DTranspose(filters=1, kernel_size=(5, 5), padding='same')(x)
        model = tf.keras.Model(inputs=zm, outputs=decoded_img)
        return model

    def lstm(self):
        zm0 = tf.keras.Input(batch_shape=(None, self.nz), name='InitialState')
        zc0 = tf.keras.Input(batch_shape=(None, None, self.nc), name='Control')
        z = layers.LSTM(self.nz, activation='linear', recurrent_activation='sigmoid',
                        use_bias=False, return_sequences=True, return_state=False,
                        name='lstm0')(zc0, initial_state=[zm0, zm0])
        model = tf.keras.Model(inputs=[zm0, zc0], outputs=z)
        return model

    def ae_lstm(self):
        initialmap = tf.keras.Input(batch_shape=(None, self.NX, self.NY, self.NZ), name='InitialMap')
        zc0 = tf.keras.Input(batch_shape=(None, None, self.nc), name='Control')
        zm0 = self.encoder()(initialmap)
        zm1 = self.lstm()([zm0, zc0])
        initialmap_ = layers.Reshape((-1, self.NX, self.NY, self.NZ), input_shape=(None, self.NX, self.NY, self.NZ))(
            initialmap)
        futuremap_ = layers.TimeDistributed(self.decoder(), name="TD_DEC")(zm1)
        output_img = layers.Concatenate(axis=1)([initialmap_, futuremap_])
        model = tf.keras.Model(inputs=[initialmap, zc0], outputs=output_img)
        opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
        model.compile(loss=["mse"], optimizer=opt)
        return model

    def ResidualLayer(self):
        pwf_mask = tf.keras.Input(batch_shape=(None, 1, self.NX, self.NY, self.NZ))
        pwf = tf.keras.Input(batch_shape=(None, None, self.NX, self.NY, self.NZ), name='WellFlowingPressure')
        output_img = tf.keras.Input(batch_shape=(None, None, self.NX, self.NY, self.NZ), name='PredictedMap')
        dt = tf.keras.Input(batch_shape=(None, None, 1), name='DifferentialTime')

        # coefficient for residual equation
        alpha = self.a * self.dx ** 2 / dt
        Cm = -(self.lam_W + self.lam_N + self.lam_E + self.lam_S + alpha)
        Wm = self.lam_W
        Em = self.lam_E
        Sm = self.lam_S
        Nm = self.lam_N
        Pm = self.gamma * self.Jw

        # initialize stencils
        stencil_n0 = tf.constant([[0, 0, 0], [0, 1, 0], [0, 0, 0]])[:, :, None,
                     None]  # [filter_height, filter_width, in_channels, out_channels]
        # stencil_n1 = tf.constant([[0,Nm,0],[Wm,Cm,Em],[0,Sm,0]])[:,:,None,None] # [filter_height, filter_width, in_channels, out_channels]
        stencil_n1 = tf.constant([[0, Em, 0], [Nm, Cm, Sm], [0, Wm, 0]])[:, :, None,
                     None]  # [filter_height, filter_width, in_channels, out_channels]
        self.stencil0 = layers.Conv2D(1, (3, 3),
                                      use_bias=False,
                                      kernel_initializer=StencilInitializer(self.stencil_n0),
                                      trainable=False, name='stencil0')
        self.stencil1 = layers.Conv2D(1, (3, 3),
                                      use_bias=False,
                                      kernel_initializer=StencilInitializer(self.stencil_n1),
                                      trainable=False, name='stencil1')

        self.stencil_n0 = stencil_n0
        self.stencil_n1 = stencil_n1

        # add image grid for no-flow boundary
        tensor_wb = layers.Lambda(lambda x: tf.pad(x, self.paddings, "SYMMETRIC"))(output_img)
        pwf_mask_wb = layers.Lambda(lambda x: tf.pad(x, self.paddings, "SYMMETRIC"))(pwf_mask)
        pwf_wb = layers.Lambda(lambda x: tf.pad(x, self.paddings, "SYMMETRIC"))(pwf)

        # prepare LHS and RHS for the residual
        tensor_t1 = self.stencil1(tensor_wb[:, 1:, :, :, :])
        tensor_t0 = self.stencil0(tensor_wb[:, :-1, :, :, :]) * alpha
        tensor_pi = self.stencil0(pwf_wb) * Pm
        tensor_pm = self.stencil0(pwf_mask_wb * tensor_wb[:, 1:, :, :, :]) * Pm
        residual = tensor_t1 + tensor_t0 + tensor_pi - tensor_pm

    def call(self, inputs):
        return self.ae_lstm()(inputs)