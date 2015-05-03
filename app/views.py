import ImageOps
import json
import os
import random
import scipy.io
from PIL import Image
import StringIO
import PIL
from django.http.response import HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import uuid
import time
import csv
import math

import numpy as np
from django.views.decorators.http import require_http_methods

from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
from pybrain.structure.modules.sigmoidlayer import SigmoidLayer
from pybrain.structure.modules.softmax import SoftmaxLayer
from pybrain.supervised import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

from ffnet import mlgraph, ffnet
import networkx as NX
import pylab

np.set_printoptions(threshold='nan', linewidth=500)

IMG_SIZE = 20
INPUT_SIZE = IMG_SIZE * IMG_SIZE
HIDDEN_LAYERS = 25
OUTPUT_LAYER = 10
LEARNING_RATE = 0.01


nn = None

def home(request):
    return render(request, 'index.html', {
        'classes': range(0, OUTPUT_LAYER)
    })


def img_to_img_data(img):
    img_data = np.asarray(img.getdata(), dtype=np.float32).reshape(img.size[1], img.size[0])
    img_data /= 256.0

    # img_data[img_data == 0] = 1
    # img_data[img_data == 255] = 0
    return img_data


def _find_bounds(img_data):
    leftmost = None
    rightmost = None
    topmost = None
    bottommost = None

    for row in img_data:
        ones = np.where(row != 0)
        if len(ones[0]) == 0:
            continue
        ones = ones[0]
        if leftmost is None or ones[0] < leftmost:
            leftmost = ones[0]
        if rightmost is None or ones[-1] > rightmost:
            rightmost = ones[-1]

    for col in img_data.T:
        ones = np.where(col != 0)
        if len(ones[0]) == 0:
            continue
        ones = ones[0]
        if topmost is None or ones[0] < topmost:
            topmost = ones[0]
        if bottommost is None or ones[-1] > bottommost:
            bottommost = ones[-1]

    return leftmost, topmost, rightmost, bottommost


def resize(img_data, size):
    h, w = img_data.shape
    wl = (size - w) / 2
    wr = size - w - wl

    ht = int(math.ceil((size - h) / 2.0))
    hb = size - h - ht

    img_data = np.vstack((np.zeros((ht, w)), img_data, np.zeros((hb, w))))
    new_h = img_data.shape[0]
    img_data = np.hstack((np.zeros((new_h, wl)), img_data, np.zeros((new_h, wr))))

    return img_data


def _transform_img(img_base64):
    img = img_base64.decode("base64")

    img = Image.open(StringIO.StringIO(img)).convert('L')
    img = ImageOps.invert(img)
    img_data = img_to_img_data(img)
    img = img.crop(_find_bounds(img_data))
    # img.thumbnail((IMG_SIZE-2, IMG_SIZE-2), PIL.Image.NEAREST)
    # img_data = img_to_img_data(img)
    # img_data = resize(img_data, IMG_SIZE)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_data = img_to_img_data(img)
    # return Image.fromarray(img_data), img_data
    return img, img_data


def _img_data_to_vec(img_data):
    img_data = img_data.ravel()
    # img_data = np.hstack(([1], img_data))
    return img_data


def _number_to_output_vector(number):
    out_vec = [0 for _ in xrange(0, OUTPUT_LAYER)]
    out_vec[number] = 1
    return out_vec


@csrf_exempt
@require_http_methods(["POST"])
def recognise(request):
    global nn
    try:
        if 'nn' in request.session:
            # nn = NetworkReader.readFrom('data/' + request.session['nn'] + '.nn')
            img = request.POST['img_data']
            img, img_data = _transform_img(img)
            print img_data
            print img_data.ravel().tolist()
            # activation = nn.activate(_img_data_to_vec(img_data))
            activation = nn.call(_img_data_to_vec(img_data))
            print activation
            return JsonResponse({
                'status': 'ok',
                'activation': activation.tolist()
            })
        else:
            resp = JsonResponse({
                'status': 'error',
                'msg': 'Run train first'
            })
            resp.status_code = 400
            return resp

    except Exception as e:
        print e
        resp = JsonResponse({
            'status': 'error',
            'msg': e
        })
        resp.status_code = 500
        return resp


def _train(X, Y, filename, epochs=50):
    global nn
    nn = buildNetwork(INPUT_SIZE, HIDDEN_LAYERS, OUTPUT_LAYER, bias=True, outclass=SoftmaxLayer)
    ds = ClassificationDataSet(INPUT_SIZE, OUTPUT_LAYER)
    for x, y in zip(X, Y):
        ds.addSample(x, y)
    trainer = BackpropTrainer(nn, ds)
    for i in xrange(epochs):
        error = trainer.train()
        print "Epoch: %d, Error: %7.4f" % (i+1, error)
    # trainer.trainUntilConvergence(verbose=True, maxEpochs=epochs, continueEpochs=10)
    if filename:
        NetworkWriter.writeToFile(nn, 'data/' + filename + '.nn')


def _train2(X, Y, filename, epochs=50):
    global nn
    conec = mlgraph((INPUT_SIZE, HIDDEN_LAYERS, OUTPUT_LAYER))
    nn = ffnet(conec)

    nn.train_momentum(X, Y, eta=0.001, momentum=0.8, maxiter=epochs, disp=True)
    # nn = Network([INPUT_SIZE, HIDDEN_LAYERS, OUTPUT_LAYER])
    # nn.SGD(zip(X, Y), epochs, 10, 3.0)



@csrf_exempt
@require_http_methods(["POST"])
def train(request):
    images = json.loads(request.POST['dataset'])
    random.shuffle(images)

    filename = str(int(time.time())) + "_" + str(uuid.uuid4().hex)
    try:
        images_data = []
        with open('data/' + filename + '.csv', 'w') as f:
            a = csv.writer(f, delimiter=',')
            for img, label in images:
                img_data = _img_data_to_vec(_transform_img(img)[1])
                img_data = np.append(img_data, [label])
                images_data.append(img_data)

            a.writerows(images_data)

        X = []
        y = []

        for d in images_data:
            X.append(d[:-1])
            y.append(_number_to_output_vector(int(d[-1])))

        _train2(X, y, filename, 10000)
        request.session['nn'] = filename

        for im in images_data:
            print im[-1], nn.call(im[:-1])

        return JsonResponse({
            'status': 'ok'
        })
    except Exception as e:
        print e
        resp = JsonResponse({
            'status': 'error',
            'msg': e.message
        })
        resp.status_code = 500
        return resp


@csrf_exempt
@require_http_methods(["POST"])
def retrain(request):
    dataset_name = request.POST['dataset_name']

    try:
        X = []
        y = []
        with open('data/' + dataset_name + '.csv', 'r') as f:
            csv_f = csv.reader(f, delimiter=',')
            for l in csv_f:
                X.append([float(ll) for ll in l[:-1]])
                y.append(_number_to_output_vector(int(float(l[-1]))))

        X = np.asarray(X)
        y = np.asarray(y)

        _train2(X, y, dataset_name, 1500)
        request.session['nn'] = dataset_name

        for im, l in zip(X, y):
            print l, nn.call(im)

        return JsonResponse({
            'status': 'ok'
        })
    except Exception as e:
        print e
        resp = JsonResponse({
            'status': 'error',
            'msg': e.message
        })
        resp.status_code = 500
        return resp


@csrf_exempt
@require_http_methods(["POST"])
def train_mnist(request):
    try:
        data = scipy.io.loadmat("mnist/ex4data1.mat")
        X = data["X"][:15000]
        y = data["y"][:15000]
        y[y == 10] = 0

        y = [_number_to_output_vector(yy) for yy in y]

        print len(X), len(y)

        f = uuid.uuid4().hex
        _train(X, y, f, 50)
        request.session['nn'] = f

        return JsonResponse({
            'status': 'ok'
        })
    except Exception as e:
        print e
        resp = JsonResponse({
            'status': 'error',
            'msg': e.message
        })
        resp.status_code = 500
        return resp

