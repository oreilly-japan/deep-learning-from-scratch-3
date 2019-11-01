# ゼロから作る Deep Learning ❸

[![pypi](https://img.shields.io/pypi/v/dezero.svg)](https://pypi.python.org/pypi/dezero)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/oreilly-japan/deep-learning-from-scratch-3/blob/master/LICENSE.md)
[![Build Status](https://travis-ci.org/koki0702/dezero.svg?branch=master)](https://travis-ci.org/koki0702/dezero)


[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/deep-learning-from-scratch-3.png" width="200px">](https://www.oreilly.co.jp/books/978487311xxxx/)

書籍『[ゼロから作るDeep Learning ❸ ―フレームワーク編](https://www.oreilly.co.jp/books/978487311xxxx/)★』(オライリー・ジャパン)のサポートサイトです。


## 概要

<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-3/images/dezero_logo.png" width="300px" />


本書では「DeZero」というディープラーニングのフレームワークを作ります。DeZeroは本書オリジナルのフレームワークです。最小限のコードで（そのすべてのコードはPython）、フレームワークのモダンな機能を実現します。本書では、この小さな、それでいて十分にパワフルなフレームワークを、全部で60★のステップで完成させます。それによって、PyTorch、TensorFlow、Chainerなどの現代のフレームワークに通じる深い知識を養います。キャッチコピーは「作るからこそ、見えるモノ」

## ファイル構成

|フォルダ名 |説明         |
|:--        |:--                  |
|dezero       |DeZeroのソースコード|
|examples     |DeZeroを使った実装例|
|steps|各ステップファイル（step01.py ~ step60.py）|
|tests|DeZeroのユニットテスト|


## 必要な外部ライブラリ

本書で使用するPytnonのバージョンと外部ライブラリは下記の通りです。

- [Python 3系](https://docs.python.org/3/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

また、オプションとして、NVIDIAのGPUで実行できる機能も提供します。その場合は、下記のライブラリが必要です。

- [CuPy](https://cupy.chainer.org/) （オプション）


## 実行方法

実行するPythonファイルは、stepsファルダにあります。
本リポジトリの一番上の階層から、もしくはstepsフォルダへ移動してから、Pythonコマンドを実行します。

```
$ python steps/step01.py

$ cd steps
$ python step31.py
```

## 正誤表

本書の正誤情報は以下のページで公開しています。

https://github.com/oreilly-japan/deep-learning-from-scratch-3/wiki/errata

本ページに掲載されていない誤植など間違いを見つけた方は、[japan＠oreilly.co.jp](<mailto:japan＠oreilly.co.jp>)までお知らせください。
