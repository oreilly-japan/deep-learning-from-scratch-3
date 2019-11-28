<p align="center">
<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/deep-learning-from-scratch-3.png" width="200px">
</p>


## フレームワーク編

本書では「DeZero」というディープラーニングのフレームワークを作ります。DeZeroは本書オリジナルのフレームワークです。最小限のコードで、フレームワークのモダンな機能を実現します。本書では、この小さな——それでいて十分にパワフルな——フレームワークを、全部で60★のステップで完成させます。それによって、PyTorch、TensorFlow、Chainerなどの現代のフレームワークに通じる深い知識を養います。

<p>
<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/dezero_logo.png" width="300px" </p>


<p>
  <a href="https://pypi.python.org/pypi/dezero"><img
		alt="pypi"
		src="https://img.shields.io/pypi/v/dezero.svg"></a>
  <a href="https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/LICENSE.md"><img
		alt="MIT License"
		src="http://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="https://travis-ci.org/oreilly-japan/deep-learning-from-scratch-3"><img
		alt="Build Status"
		src="https://travis-ci.org/oreilly-japan/deep-learning-from-scratch-3.svg?branch=master"></a>
</p>

## ファイル構成

|フォルダ名 |説明         |
|:--        |:--                  |
|[dezero](/dezero)       |DeZeroのソースコード|
|[examples](/examples)     |DeZeroを使った実装例|
|[steps](/steps)|各stepファイル（step01.py ~ step60.py）|
|[tests](/tests)|DeZeroのユニットテスト|

- 各stepファイルの概要については[「stepファイルの説明」](../../wiki/Step%E3%83%95%E3%82%A1%E3%82%A4%E3%83%AB%E3%81%AE%E8%AA%AC%E6%98%8E)をご覧ください。

- 本書のステップ35で行う実験結果は[「tanhの高階微分」](https://github.com/oreilly-japan/deep-learning-from-scratch-3/tree/tanh)をご覧ください。

## 必要な外部ライブラリ

本書で使用するPytnonのバージョンと外部ライブラリは下記の通りです。

- [Python 3系](https://docs.python.org/3/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

またオプションとして、NVIDIAのGPUで実行できる機能も提供します。その場合は下記のライブラリが必要です。

- [CuPy](https://cupy.chainer.org/) （オプション）


## 実行方法

本書で説明するPythonファイルは、主に[steps](/steps)ファルダにあります。
実行するためには、本リポジトリの一番上の階層からPythonコマンドを実行します。


```
# 👍(Good)
$ python steps/step01.py
$ python steps/step31.py

# ❌(NG)
$ cd steps
$ python step31.py
```

## DeZeroの実装例

他のDeZeroの実装例は[examples](/examples)にあります。

[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/example_tanh.png" height="175"/>](/examples/tanh.py)[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/example_spiral.png" height="175"/>](/examples/spiral.py)[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/example_gpu.png" height="175"/>](https://colab.research.google.com/github/oreilly-japan/deep-learning-from-scratch-3/blob/master/examples/mnist_colab_gpu.ipynb)

## 正誤表

本書の正誤情報は、[「:mag_right: 正誤表ページ」](../../wiki/Errata)に掲載しています。

正誤表ページに掲載されていない誤植や間違いなどを見つけた方は、[:email: japan＠oreilly.co.jp](<mailto:japan＠oreilly.co.jp>)までお知らせください。
