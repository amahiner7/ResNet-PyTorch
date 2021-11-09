import glob
from model.ResNet import ResNet
from config.file_path import *
from config.model_parameters import *
from config.hyper_parameters import *
from data.load_data import load_data
from utils.common import *

LOAD_MODEL_FILE_DIR = ""


def load_model():
    save_model_file_names = sorted(glob.glob(os.path.join(LOAD_MODEL_FILE_DIR, "*.pth")))

    if len(save_model_file_names) > 0:
        model = ResNet(input_size=(IMAGE_CHANNEL, IMAGE_WIDTH, IMAGE_HEIGHT),
                       params=RESNET50_MODEL_PARAMS,
                       device='cpu')
        last_file = save_model_file_names[len(save_model_file_names) - 1]
        model.load(last_file)
        model.summary()

        return model
    else:
        raise Exception("It can't find model files.")


_, _, test_data_loader = load_data()
resnet_model = load_model()

num_rows = 5
num_cols = 3
num_samples = num_rows * num_cols

for data, label in test_data_loader:
    for index in range(num_samples):
        input_tensor = data[index].unsqueeze(0)
        image = data[index].numpy()
        image = (image / np.amax(image) + 1)
        image = image / np.amax(image)
        image = np.transpose(image, (1, 2, 0))
        true_label = label[index].numpy()

        plt.subplot(num_rows, num_cols, index + 1)

        predict = resnet_model(input_tensor)
        predict = (predict.squeeze())

        plot_image(pred=predict, label=true_label, image=image)
    break

plt.tight_layout()
plt.show()
