import comet_ml 

# Data loaders
from data_loader.conv_atac_data_loader import  ConvATACseqDataLoader



# Models
from models.conv_atac_model import ConvATACseqModel
from models.lstm_atac_model import LSTMATACseqModel
from models.diluted_atac_model import DillutedATACseqModel

# Model trainers
from trainers.conv_atac_trainer import ConvATACseqModelTrainer
from trainers.lstm_atac_trainer import LSTMATACseqModelTrainer
from trainers.diluted_atac_trainer import DillutedATACseqModelTrainer

from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args
from rich.console import Console
from pathlib import  Path


console = Console()

def selectModel(model):

    loader = ConvATACseqDataLoader

    if model == "Conv":
        model = ConvATACseqModel
        trainer = ConvATACseqModelTrainer

    elif  model == 'lstm':
        model = LSTMATACseqModel
        trainer = LSTMATACseqModelTrainer

    elif model == 'dillution':
        model = DillutedATACseqModel
        trainer = DillutedATACseqModelTrainer

    return loader, model, trainer




def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        console.print("missing or invalid arguments")
        exit(0)

   # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    loader, model, trainer = selectModel(args.model)

    #console.print('Create the data generator.')
    data_loader = loader(config)

    console.print('Create the model ....')
    model = model(config)

    console.print('Create the trainer ...')
    trainer = trainer(model.model, 
                      data_loader.get_train_data(), 
                      data_loader.get_test_data(), 
                      config)

    console.print('Start training the model ...')
    trainer.train()

    console.print("Save model ...")
    
    Path(config.model_dir).mkdir()
    fout = Path(config.model_dir) / config.model.name
    trainer.model.save(fout)


if __name__ == '__main__':
    main()
