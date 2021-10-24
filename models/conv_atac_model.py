from base.base_model import BaseModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, AveragePooling1D, Input
from utils.PWMInitializer import PWMInitializer

class ConvATACseqModel(BaseModel):
    def __init__(self, config):
        super(ConvATACseqModel, self).__init__(config)
        self.build_model()

    def build_model(self):


        if self.config.model.use_pwminit:
            kinit = PWMInitializer(pwm_file=self.config.data.pwm, max_motifs=20* self.config.model.nb_filters)
        else:
            kinit = "glorot_uniform"

        input_seq = Input((self.config.model.seq_len,4),name = 'input_sequence')

        conv1 = Conv1D(filters= self.config.model.nb_filters,
                              kernel_initializer= kinit,
                              bias_initializer="zeros",
                              kernel_size=self.config.model.kern_size,
                              kernel_regularizer='l2',
                              padding="same",
                              activation="relu")(input_seq)

        #conv1 = Conv1D(filters= round(self.config.model.nb_filters/2), kernel_size=self.config.model.kern_size,activation="relu")(input_seq)

        avg_pool1 = AveragePooling1D( self.config.model.pooling_size)(conv1)

        flat = Flatten()(avg_pool1)

        net = Dense(units = self.config.model.nb_hidden,
                    activation = "relu")(flat)
        net = Dropout(self.config.model.dropout_rate)(net)

        net = Dense(units = 1,
                    activation = 'sigmoid')(net)

        self.model = Model(inputs = input_seq, outputs = net)

        self.model.compile(
              loss=self.config.model.loss,
              optimizer=self.config.model.optimizer,
              metrics=['accuracy'])

        self.model.summary()
