# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""

# file kfkd.py
import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'

def load(test=False, cols=None):
    """testがTrueの場合はFTESTからデータを読み込み、Falseの場合はFTRAINから読み込みます。
    colsにリストが渡された場合にはそのカラムに関するデータのみ返します。
    """

    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname)) # pandasのdataframeを使用

    # スペースで句切られているピクセル値をnumpy arrayに変換
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # カラムに関連するデータのみを抽出
        df = df[list(cols) + ['Image']]

    print(df.count())  # カラム毎に値が存在する行数を出力
    df = df.dropna()  # データが欠けている行は捨てる

    X = np.vstack(df['Image'].values) / 255.  # 0から1の値に変換
    X = X.astype(np.float32)

    if not test:  # ラベルが存在するのはFTRAINのみ
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # -1から1の値に変換
        X, y = shuffle(X, y, random_state=42)  # データをシャッフル
        y = y.astype(np.float32)
    else:
        y = None

    return X, y


###
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt

def build_model01():
    model = Sequential()
    model.add(Dense(100, input_dim=9216))
    model.add(Activation('relu'))
    model.add(Dense(30))

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)
    hist = model.fit(X, y, nb_epoch=100, validation_split=0.2)
    
        
    plt.plot(hist.history['loss'], linewidth=3, label='train')
    plt.plot(hist.history['val_loss'], linewidth=3, label='valid')
    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(1e-3, 1e-2)
    plt.yscale('log')
    plt.show()


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)





if __name__ == "__main__":
    X, y = load()
    print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
        X.shape, X.min(), X.max()))
    print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
        y.shape, y.min(), y.max()))
    
    build_model01()
    
    json_string = model.to_json()
    open('model1_architecture.json', 'w').write(json_string)
    model.save_weights('model1_weights.h5')