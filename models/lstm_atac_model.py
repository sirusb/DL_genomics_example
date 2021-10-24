from base.base_model import BaseModel
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, AveragePooling1D, Input, LSTM, Bidirectional

class LSTMATACseqModel(BaseModel):
    def __init__(self, config):
        super(LSTMATACseqModel, self).__init__(config)
        print(self.config.model)
        self.build_model()

    def build_model(self):
        input_seq = Input((self.config.model.seq_len,4),name = 'input_sequence')
        conv1 = Conv1D(filters= self.config.model.nb_filters,
                       kernel_size= self.config.model.kern_size,
                       padding= "same",
                       activation= "relu")(input_seq)

        lstm = Bidirectional(LSTM(units= self.config.model.seq_len))(conv1)

        net = Dense(units = self.config.model.nb_hidden,
                    activation = "relu")(lstm)
        net = Dropout(self.config.model.dropout_rate)(net)
        net = Dense(units = 1,
                    activation = 'sigmoid')(net)

        self.model = Model(inputs = input_seq, outputs = net)

        self.model.compile(
              loss=self.config.model.loss,
              optimizer=self.config.model.optimizer,
              metrics=['accuracy'])
        self.model.summary()

