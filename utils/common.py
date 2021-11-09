import numpy as np
import datetime
import matplotlib.pyplot as plt


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


# 정확도 계산 함수
def flat_accuracy(output, labels):
    pred_flat = np.argmax(output, axis=1).flatten()
    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def metric_batch(output, labels):
    prediction = output.argmax(1, keepdim=True)
    corrects = prediction.eq(labels.view_as(prediction)).sum().item()
    return corrects


# 시간 표시 함수
def format_time(elapsed):
    # 반올림
    elapsed_rounded = int(round(elapsed))

    return str(datetime.timedelta(seconds=elapsed_rounded))


def display_loss(history):
    train_loss = history['loss']
    val_loss = history['val_loss']

    # 그래프로 표현
    x_len = np.arange(len(train_loss))
    plt.figure()
    plt.plot(x_len, val_loss, marker='.', c="blue", label='Validation loss')
    plt.plot(x_len, train_loss, marker='.', c="red", label='Train loss')
    # 그래프에 그리드를 주고 레이블을 표시
    plt.legend(loc='upper right')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.show()

    if history.get('learning_rate') is not None:
        learning_rate = history['learning_rate']
        x_len = np.arange(len(learning_rate))
        plt.clf()
        plt.figure()
        plt.plot(x_len, learning_rate, marker='.', c="yellow", label='Learning rate')
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('Learning rate')
        plt.title('Learning rate')
        plt.show()


def plot_image(pred, label, image):
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image)

    if np.math.fabs(pred - float(label)) < 10.0:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("Pred: {:.1f} | Label: {}".format(pred, label), color=color)