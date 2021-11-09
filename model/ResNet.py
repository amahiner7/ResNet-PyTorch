import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torchsummary import summary
from tensorboardX import SummaryWriter
import time
import math

from model.Conv1_Block import Conv1_Block
from model.ResidualBlock import ResidualBlock
from config.hyper_parameters import *
from config.file_path import *
from utils.common import *
from model.custom_scheduler.CosineAnnealingWarmRestarts import CosineAnnealingWarmUpRestarts


class ResNet(nn.Module):
    def __init__(self, input_size, params, use_lambda_lr=True, device=None, name='ResNet50'):
        super().__init__()

        self.input_size = input_size
        self.params = params
        self.device = device
        self.name = name

        self._make_layers()
        self.criterion = None
        self.optimizer = None
        self.lr_scheduler = None
        self.use_lambda_lr = use_lambda_lr
        self._initialize_weights()
        self._set_cuda_environment(device)

    def _make_layers(self):
        self.conv1_block = Conv1_Block(in_channels=self.input_size[0])
        self.max_pooling2d_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        in_channels = self.conv1_block.out_channels

        self.residual_block_list = nn.ModuleList()
        for block_index in range(len(self.params)):
            residual_block = ResidualBlock(in_channels=in_channels, params=self.params[block_index])
            self.residual_block_list.append(residual_block)
            in_channels = residual_block.out_channels

        self.avg_pool2d = nn.AdaptiveAvgPool2d(1)
        self.feature_layer = nn.Linear(in_features=in_channels, out_features=256)
        in_channels = 256
        self.feature_layer_activation = nn.ReLU()
        self.output_layer = nn.Linear(in_features=in_channels, out_features=1)
        self.output_layer_activation = nn.ReLU()

    def _set_summary_writer(self, tensorboard_dir):
        self.loss_writer = SummaryWriter(tensorboard_dir)

    def _set_cuda_environment(self, device):
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
                torch.cuda.manual_seed_all(SEED_VALUE)
            else:
                self.device = 'cpu'

        print(f"{self.name} : {self.device} is available.")
        self.to(self.device)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def _check_compile(self):
        if self.criterion is None or self.optimizer is None or self.lr_scheduler is None:
            if self.use_lambda_lr:
                criterion = torch.nn.MSELoss().to(self.device)
                optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
                lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch:
                1.0 if epoch < 10 else np.math.exp(0.1 * (10 - epoch)))
            else:
                criterion = torch.nn.MSELoss().to(self.device)
                optimizer = torch.optim.Adam(self.parameters(), lr=0)
                lr_scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer, T_0=20, T_mult=1,
                                                             eta_max=LEARNING_RATE,
                                                             T_up=10,
                                                             gamma=0.2)

            self.compile(criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler)

    def _get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def compile(self, criterion, optimizer, lr_scheduler):
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward(self, inputs):
        x = self.conv1_block(inputs)
        x = self.max_pooling2d_1(x)

        for block_index in range(len(self.residual_block_list)):
            residual_block = self.residual_block_list[block_index]
            x = residual_block(x)

        x = self.avg_pool2d(x)
        x = x.view(x.size(0), -1)
        x = self.feature_layer(x)
        x = self.feature_layer_activation(x)
        x = self.output_layer(x)
        x = self.output_layer_activation(x)

        return x

    def train_on_batch(self, data_loader, log_interval=1):
        loss_list = []
        complete_batch_size = 0

        self._check_compile()
        self.train()

        for batch_index, data in enumerate(data_loader):
            input = data[0].to(self.device)
            label = data[1].to(self.device).unsqueeze(dim=-1)

            # Forward 실행
            predict = self.forward(input)

            # Gradient 초기화
            self.optimizer.zero_grad()

            # Loss 계산
            loss = self.criterion(predict, label)

            # Back-propagation 수행
            loss.backward()

            # Weight 업데이트
            self.optimizer.step()

            loss_list.append(loss.item())

            complete_batch_size += len(input)
            if (batch_index % log_interval == 0 or (batch_index + 1) == len(data_loader)) and batch_index != 0:
                print(" BATCH: [{}/{}({:.0f}%)] | TRAIN LOSS: {:.4f}".format(
                    complete_batch_size,
                    len(data_loader.dataset),
                    100.0 * (batch_index + 1) / len(data_loader),
                    loss.item()))

        return loss_list

    def evaluate(self, data_loader):
        self.eval()
        loss_list = []

        with torch.no_grad():
            for batch_index, data in enumerate(data_loader):
                input = data[0].to(self.device)
                label = data[1].to(self.device).unsqueeze(dim=-1)

                predict = self.forward(input)
                loss = self.criterion(predict, label)
                loss_list.append(loss.item())

        return loss_list

    def train_on_epoch(self, train_data_loader, valid_data_loader, epochs,
                       tensorboard_dir=TENSORBOARD_LOG_DIR, log_interval=1):
        best_val_loss = math.inf
        loss_history = []
        val_loss_history = []
        learning_rate_history = []

        self._check_compile()
        self._set_summary_writer(tensorboard_dir)

        for epoch in range(epochs):
            print("=============== TRAINING EPOCHS {} / {} ===============".format(epoch + 1, epochs))
            train_start_time = time.time()

            train_loss_list = self.train_on_batch(data_loader=train_data_loader, log_interval=log_interval)
            val_loss_list = self.evaluate(data_loader=valid_data_loader)

            # Learning rate 업데이트
            # self.lr_scheduler.step(epoch=epoch)
            self.lr_scheduler.step()

            learning_rate = self._get_lr()

            train_loss = np.mean(train_loss_list)
            val_loss = np.mean(val_loss_list)
            if val_loss < best_val_loss:
                model_file_path = self.save(model_file_dir=MODEL_FILE_DIR,
                                            model_file_name=MODEL_FILE_NAME,
                                            epoch=epoch,
                                            val_loss=val_loss)
                print("val_loss improved from {:.5f} to {:.5f}, saving model to ".format(
                    best_val_loss, val_loss) + model_file_path)
                best_val_loss = val_loss
            else:
                print("val_loss did not improve from {:.5f}".format(best_val_loss))

            print("TRAINING ELAPSED TIME: {} | TRAIN LOSS: {:.4f} | VAL LOSS: {:.4f} | LEARNING RATE: {}\n".
                  format(format_time(time.time() - train_start_time), train_loss, val_loss, learning_rate))

            loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            learning_rate_history.append(learning_rate)
            self.loss_writer.add_scalars('history',
                                         {'train': train_loss, 'validation': val_loss},
                                         epoch+1)
            self.loss_writer.add_scalar('history/learning_rate', learning_rate, epoch+1)

        self.loss_writer.close()

        history = {'loss': loss_history, 'val_loss': val_loss_history, 'learning_rate': learning_rate_history}

        return history

    def save(self, model_file_dir, model_file_name, epoch, val_loss):
        if not os.path.exists(model_file_dir):
            os.mkdir(model_file_dir)
            print("Directory: {} is created.".format(model_file_dir))

        model_file_name = model_file_name.format(epoch + 1, val_loss)
        model_file_path = os.path.join(model_file_dir, model_file_name)
        torch.save({'net': self.state_dict(), 'optim': self.optimizer.state_dict()}, model_file_path)

        return model_file_path

    def load(self, model_file_path):
        dict_model = torch.load(model_file_path, map_location=self.device)
        self._check_compile()
        self.load_state_dict(dict_model['net'])
        self.optimizer.load_state_dict(dict_model['optim'])

    def summary(self):
        summary(self, input_size=self.input_size, device=self.device)
