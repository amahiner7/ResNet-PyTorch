from config.model_parameters import *
from config.hyper_parameters import *
from model.ResNet import ResNet
from data.load_data import load_data
from utils.common import *

train_data_loader, valid_data_loader, _ = load_data()

model = ResNet(input_size=(IMAGE_CHANNEL, IMAGE_WIDTH, IMAGE_HEIGHT), params=RESNET50_MODEL_PARAMS, use_lambda_lr=False)
model.summary()

if __name__ == '__main__':
    history = model.train_on_epoch(train_data_loader=train_data_loader,
                                   valid_data_loader=valid_data_loader,
                                   epochs=NUM_EPOCHS,
                                   log_interval=30)

    display_loss(history)
