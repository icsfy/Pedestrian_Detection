# HOG+SVM进行行人检测
> 运行环境为：Ubuntu 16.04 && OpenCV 2.4.13  
> 代码参考自：http://blog.csdn.net/masibuaa/article/details/16105073  
> INRIA Person数据库下载：
【[相关说明](http://pascal.inrialpes.fr/data/human/)】  
>【直接下载地址（970M）】ftp://ftp.inrialpes.fr/pub/lear/douze/data/INRIAPerson.tar

## 开始前的准备工作
开始前建立文件夹用于存储正负样本和HardExample，正样本图片直接复制INRIA中的正样本图片，负样本图片通过裁剪得到。
```shell
$ mkdir -p dataset/pos dataset/neg dataset/HardExample
$ cp INRIAPerson/96X160H96/Train/pos/* dataset/pos/
```
## 编译出可执行文件
```shell
$ cmake .
$ make
```
## 第一步，裁剪出负样本图片
INRIA中有1218张负样本图片，`CropImage`从每一张图片中随机裁剪出10张大小为64x128的图片，最终总共会得到12180张图片，存储在dataset/neg文件夹中。上面已经编译出了可执行文件，直接通过`CropImage`裁剪出负样本图片。
```shell
$ ./CropImage
```
## 第二步，使用正负样本进行训练
先修改 `dataset.h` 里面参数，将 `TRAIN` 由 `false` 改为 `true` , 以进行训练，参数修改后需通过 `make` 重新编译可执行文件，然后通过 `SvmTrainUseHog` 开始训练。
```shell
$ make
$ ./SvmTrainUseHog
```
 到这里已经得到了 `SVM_HOG.xml` 可以用来进行检测，但是检测效果不太好，所以下面加入 HardExample 来进行进一步训练。

## 第三步，得到HardExample
通过 `GetHardExample` 从 INRIA 原始的负样本图片中检测出 HardExample ，图片会保存到 dataset/HardExample
```shell
$ ./GetHardExample
```
## 第四步，将正负样本和HardExample一起重新进行训练
将 HardExample 图片列表写入文件 `HardExample_FromINRIA_NegList.txt` ，
修改 `dataset.h` 里面的参数，将 `HardExampleNO` 由 `0` 改为第三步中得到的图片数目，修改后通过 `make` 重新编译可执行文件，最后通过 `SvmTrainUseHog` 重新训练。
```shell
$ ls dataset/HardExample/ >HardExample_FromINRIA_NegList.txt
$ make
$ ./SvmTrainUseHog
```
## 至此训练完成。
检测示例图片：  
![检测示例图片](https://github.com/icsfy/Pedestrian_Detection/raw/master/ImgProcessed.jpg)
## 其它说明
* `SVM_HOG.xml`为最终训练好的SVM分类器
* `ImageDetect`可对图片进行检测
* `VideoDetect`可对视频进行检测
* `PeopleDetect`为OpenCV默认参数的行人检测程序

## [More](https://github.com/icsfy/Pedestrian_Detection/MORE.md)
