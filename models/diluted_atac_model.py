from base.base_model import BaseModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, AveragePooling1D, Input, concatenate

class DillutedATACseqModel(BaseModel):
    def __init__(self, config):
        super(DillutedATACseqModel, self).__init__(config)
        print(self.config.model)
        self.build_model()

    def build_model(self):
        input_seq = Input((self.config.model.seq_len,4),name = 'input_sequence')

        conv1 = Conv1D(filters= self.config.model.nb_filters,
                       kernel_size= self.config.model.kern_size,
                       padding= "same",
                       activation= "relu")(input_seq)

        pools = []
        for i in range(self.config.model.nb_dilutted):
            dilution_conv = Conv1D(filters= self.config.model.nb_dillution_filters,
                                   kernel_size= self.config.model.dillution_ksize,
                                   activation='relu',
                                   dilation_rate= self.config.model.dilution_rate * (i+1),
                                   padding='same')(conv1)

            pool = AveragePooling1D(self.config.model.pooling_size)(dilution_conv)
            pools.append(pool)

        mrged_layers = concatenate(pools)
        flat = Flatten()(mrged_layers)

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

