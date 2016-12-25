#### 开始前建立文件夹用于存储正负样本和HardExample，并复制正样本图片
```shell
mkdir -p dataset/pos dataset/neg dataset/HardExample
cp INRIAPerson/96X160H96/Train/pos/* dataset/pos/
```
#### 编译出可执行文件
```shell
cmake .
make
```
### 第一步，通过CropImage裁剪出负样本图片
```shell
./CropImage
```
### 第二步，使用正负样本进行训练
#### 先修改 dataset.h 里面参数，将 TRAIN 由 false 改为 true, 以进行训练，修改后通过 make 重新编译可执行文件
```shell
make
./SvmTrainUseHog
```
##### 到这里已经得到了 SVM_HOG.xml 可以进行检测，但效果不太好，所以下面加入 HardExample 进行进一步训练

### 第三步，检测得到HardExample
```shell
./GetHardExample
```
### 第四步，将正负样本和HardExample一起重新训练
#### HardExample图片列表写入HardExample_FromINRIA_NegList.txt
#### 修改 dataset.h 参数，将 HardExampleNO 由 0 改为第三步中得到的图片数目，修改后通过 make 重新编译可执行文件
```shell
ls dataset/HardExample/ >HardExample_FromINRIA_NegList.txt
make
./SvmTrainUseHog
```
#### 至此训练完成。

#### ImageDetect对图片进行检测，VideoDetect对视频进行检测。
