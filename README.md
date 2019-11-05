# ゼロから作る Deep Learning ❸

[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/deep-learning-from-scratch-3.png" width="200px">](https://www.oreilly.co.jp/books/978487311xxxx/)

書籍『[ゼロから作るDeep Learning ❸ ―フレームワーク編](https://www.oreilly.co.jp/books/978487311xxxx/)★』(オライリー・ジャパン発行)のサポートサイトです。本書籍で使用するソースコードがまとめられています。


## フレームワーク編

<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/dezero_logo.png" width="300px" />

[![pypi](https://img.shields.io/pypi/v/dezero.svg)](https://pypi.python.org/pypi/dezero)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/LICENSE.md)
[![Build Status](https://travis-ci.org/oreilly-japan/deep-learning-from-scratch-3.svg?branch=master)](https://travis-ci.org/oreilly-japan/deep-learning-from-scratch-3)


本書では「DeZero」というディープラーニングのフレームワークを作ります。DeZeroは本書オリジナルのフレームワークです。最小限のコードで、フレームワークのモダンな機能を実現します。本書では、この小さな——それでいて十分にパワフルな——フレームワークを、全部で60★のステップで完成させます。それによって、PyTorch、TensorFlow、Chainerなどの現代のフレームワークに通じる深い知識を養います。

## ファイル構成

|フォルダ名 |説明         |
|:--        |:--                  |
|[dezero](/dezero)       |DeZeroのソースコード|
|[examples](/examples)     |DeZeroを使った実装例|
|[steps](/steps)|各stepファイル（step01.py ~ step60.py）|
|[tests](/tests)|DeZeroのユニットテスト|

stepファイルの説明は、[:mag_right: [wikiページ] ](../../wiki/Step-files)にまとめています。


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

## 正誤表

本書の正誤情報は、[:mag_right: [正誤表ページ] ](../../wiki/Errata)に掲載しています。

正誤表ページに掲載されていない誤植や間違いなどを見つけた方は、[:email: japan＠oreilly.co.jp](<mailto:japan＠oreilly.co.jp>)までお知らせください。
