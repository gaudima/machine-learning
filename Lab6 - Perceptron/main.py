from struct import unpack
from PIL import Image
from PIL import ImageOps
from PyQt5.QtGui import QGuiApplication, QImage
from PyQt5.QtQml import QQmlApplicationEngine, qmlRegisterType
from PyQt5.QtCore import *

import sys
import numpy as np
from PyQt5.QtQuick import QQuickPaintedItem


def read_idx1(file):
    with open(file, 'rb') as f:
        magic, length = unpack('>ii', f.read(8))
        if magic != 2049:
            raise RuntimeError('wrong file format')
        return np.fromfile(f, np.uint8, length).astype(np.int64)


def read_idx3(file):
    data = []
    with open(file, 'rb') as f:
        magic, length = unpack('>ii', f.read(8))
        if magic != 2051:
            raise RuntimeError('wrong file format')
        rows, columns = unpack('>ii', f.read(8))
        for i in range(length):
            img = np.fromfile(f, np.uint8, rows * columns).astype(np.float)
            img = img / 255
            data.append(img)
    return np.array(data)


def resize_image(data):
        img = Image.new('L', (28, 28))
        img.putdata(data)
        img = img.resize((12, 12), Image.ANTIALIAS)
        return np.array(img.getdata())


def create_dumb_network(input_count, hidden_count, output_count):
    scale = 1.0 / input_count ** (1 / 2)
    input2hidden = np.random.normal(0, scale, (input_count, hidden_count))
    hidden2output = np.random.uniform(size=(hidden_count, output_count)) / np.sqrt(hidden_count)
    delta0 = np.zeros((input_count, hidden_count)).astype(np.float)
    delta1 = np.zeros((hidden_count, output_count)).astype(np.float)
    return [input2hidden, hidden2output, delta0, delta1]


def sigmoid(x):
        return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def get_output(network, input):
    hidden_layer = tanh(np.dot(network[0].T, input))
    output_layer = sigmoid(np.dot(network[1].T, hidden_layer))
    return hidden_layer, output_layer


def correct_weights(network, input, output_layers, expected):
    hidden, output = output_layers
    # print(hidden)
    # print('output  ', output)
    # print('expected', expected)
    delta_output = output * (1 - output) * (expected - output)

    error = np.dot(network[1], delta_output)
    delta_hidden = (1 - error * error) * error

    # network[3] = delta_output * np.reshape(output_layers[0], (output_layers[0].shape[0], 1))
    # network[2] = delta_hidden * np.reshape(input, (input.shape[0], 1))
    network[3] = delta_output * np.reshape(output_layers[0], (output_layers[0].shape[0], 1))
    network[2] = delta_hidden * np.reshape(input, (input.shape[0], 1))
    eta = 0.01
    network[0] += eta * network[2]
    network[1] += eta * network[3]
    # print(network)


def train_step(network, label, data):
    output = get_output(network, data)
    expected = np.array([1 if k == label else 0 for k in range(len(network[1][0]))]).astype(np.float)
    correct_weights(network, data, output, expected)
    return output[1], expected


def train_generation(network, train_labels, train_data):
    for idx, label in enumerate(train_labels):
        print('step:', idx)
        train_step(network, label, train_data[idx])


def train(network, train_labels, train_data):
    for i in range(1):
        # print('generation:', i)
        train_generation(network, train_labels, train_data)


def test(network, test_labels, test_data):
    for idx, label in enumerate(test_labels):
        test_step(network, label, test_data[idx])


def test_step(network, label, data):
    output = get_output(network, data)
    expected = np.array([1 if k == label else 0 for k in range(len(network[1][0]))]).astype(np.float)
    return output[1], expected


def data2image(data):
    data = data * 255
    data = data.astype(np.uint8)
    img = Image.new('L', (12, 12))
    img.putdata(data)
    img = ImageOps.invert(img)
    return img


class TrainGui(QObject):
    def __init__(self, parent=None):
        QObject.__init__(self, parent=parent)
        self.train_data = read_idx3('mnist/train.idx3-ubyte')
        self.train_labels = read_idx1('mnist/train.idx1-ubyte')
        self.network = create_dumb_network(28 * 28, 100, 10)
        self._index = -1
        self._paused = True
        self._timer = QTimer(parent=self)
        self._timer.timeout.connect(self.nextStep)
        self.nextStep()

    indexChanged = pyqtSignal()

    @pyqtProperty('int', notify=indexChanged)
    def index(self):
        return self._index

    pausedChanged = pyqtSignal()

    @pyqtProperty('bool', notify=pausedChanged)
    def paused(self):
        return self._paused

    @paused.setter
    def paused(self, spaused):
        self._paused = spaused
        if self._paused:
            self._timer.stop()
        else:
            self._timer.start()
        self.pausedChanged.emit()

    resultChanged = pyqtSignal()

    @pyqtProperty('QString', notify=resultChanged)
    def result(self):
        return self._result

    expectedChanged = pyqtSignal()

    @pyqtProperty('QString', notify=expectedChanged)
    def expected(self):
        return self._expected

    imageChanged = pyqtSignal()

    @pyqtProperty('QImage', notify=imageChanged)
    def image(self):
        data = self.train_data[self._index]
        data *= 255
        img = QImage(data.astype(np.uint8), 28, 28, QImage.Format_Grayscale8)
        img.invertPixels()
        return img

    @pyqtSlot()
    def nextStep(self):
        self._index += 1
        if self._index >= len(self.train_labels):
            self._index -= 1
            self.paused = True
            return
        self._result, self._expected = train_step(self.network,
                                                  self.train_labels[self._index],
                                                  self.train_data[self._index])
        frm = {'float_kind': lambda x: "{0:.4f}".format(x)}
        self._result = np.array2string(self._result, formatter=frm)
        self._expected = np.array2string(self._expected, formatter=frm)
        self.resultChanged.emit()
        self.expectedChanged.emit()
        self.indexChanged.emit()
        self.imageChanged.emit()


class TestGui(QObject):
    def __init__(self, network, parent=None):
        QObject.__init__(self, parent=parent)
        self.test_data = read_idx3('mnist/test.idx3-ubyte')
        self.test_labels = read_idx1('mnist/test.idx1-ubyte')
        self.network = network
        self._index = -1
        self._paused = True
        self._classified_right = 0
        self._timer = QTimer(parent=self)
        self._timer.timeout.connect(self.nextStep)
        self._expectednum = 0
        self._resultnum = 0
        self._result = ''
        self._expected = ''
        # self.nextStep()

    indexChanged = pyqtSignal()

    @pyqtProperty('int', notify=indexChanged)
    def index(self):
        return self._index

    pausedChanged = pyqtSignal()

    @pyqtProperty('bool', notify=pausedChanged)
    def paused(self):
        return self._paused

    @paused.setter
    def paused(self, spaused):
        self._paused = spaused
        if self._paused:
            self._timer.stop()
        else:
            self._timer.start()
        self.pausedChanged.emit()

    resultChanged = pyqtSignal()

    @pyqtProperty('QString', notify=resultChanged)
    def result(self):
        return self._result

    expectedChanged = pyqtSignal()

    @pyqtProperty('QString', notify=expectedChanged)
    def expected(self):
        return self._expected

    @pyqtProperty('int', notify=resultChanged)
    def resultnum(self):
        return self._resultnum

    @pyqtProperty('int', notify=expectedChanged)
    def expectednum(self):
        return self._expectednum

    imageChanged = pyqtSignal()

    @pyqtProperty('QImage', notify=imageChanged)
    def image(self):
        if self._index < 0:
            return QImage()
        data = self.test_data[self._index]
        data *= 255
        img = QImage(data.astype(np.uint8), 28, 28, QImage.Format_Grayscale8)
        img.invertPixels()
        return img

    @pyqtSlot()
    def nextStep(self):
        self._index += 1
        if self._index >= len(self.test_labels):
            self._index -= 1
            self.paused = True
            return
        self._result, self._expected = test_step(self.network,
                                                 self.test_labels[self._index],
                                                 self.test_data[self._index])
        frm = {'float_kind': lambda x: "{0:.4f}".format(x)}
        self._expectednum = np.argmax(self._expected)
        self._resultnum = np.argmax(self._result)
        self._result = np.array2string(self._result, formatter=frm)
        self._expected = np.array2string(self._expected, formatter=frm)
        if self._expectednum != self._resultnum:
            # self.paused = True
            print(self._classified_right / (self.index + 1))
        else:
            self._classified_right += 1
        self.resultChanged.emit()
        self.expectedChanged.emit()
        self.indexChanged.emit()
        self.imageChanged.emit()

    @pyqtSlot()
    def prevStep(self):
        self._index -= 1
        if self._index < 0:
            self._index += 1
        self.indexChanged.emit()
        self.imageChanged.emit()

class QImagePainter(QQuickPaintedItem):
    def __int__(self, parent=None):
        QQuickPaintedItem.__init__(self, parent=parent)
        self._image = QImage(28, 28, QImage.Format_Grayscale8)

    imageChanged = pyqtSignal()

    @pyqtProperty('QImage', notify=imageChanged)
    def image(self):
        return self._image

    @image.setter
    def image(self, simage):
        self._image = simage
        self.imageChanged.emit()
        self.update()

    def paint(self, painter):
        painter.drawImage(QPoint(0, 0), self._image.scaled(self.width(), self.height()))


if __name__ == '__main__':
    # train_data = read_idx3('mnist/train.idx3-ubyte')
    # train_labels = read_idx1('mnist/train.idx1-ubyte')
    #
    # test_data = read_idx3('mnist/test.idx3-ubyte')
    # test_labels = read_idx1('mnist/test.idx1-ubyte')
    #
    # network = create_dumb_network(28 * 28, 100, 10)
    #
    # train(network, train_labels, train_data)
    # print('===== TEST =====')
    # test(network, test_labels, test_data)
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine(parent=app)
    ctx = engine.rootContext()
    trainGui = TrainGui()
    ctx.setContextProperty("trainGui", trainGui)
    testGui = TestGui(trainGui.network)
    ctx.setContextProperty("testGui", testGui)
    qmlRegisterType(QImagePainter, "PerceptronImagePainter", 1, 0, "ImagePainter")
    engine.load(QUrl("gui/main.qml"))
    exit(app.exec_())
