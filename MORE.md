# 摘要（Abstract）
...
# 主要内容

## 介绍（Introduction）
...
## 研究现状（Related work）
...
## 技术方案（Technique）
 * HOG特征提取+SVM分类器训练、 代码采用OpenCV实现  
 * OpenCV中已经实现了HOG特征提取和SVM分类器相关算法。我们直接调用相关函数即可。   
 * 行人数据库采用 INRIA数据库。  
  > 该数据库是目前使用最多的静态行人检测数据库，提供原始图片及相应的标注文件。训练集有正样本614张（包含2416个行人），负样本1218张；测试集有正样本288张（包含1126个行人），负样本453张。图片中人体大部分为站立姿势且高度大于100个象素，部分标注可能不正确。图片主要来源于GRAZ-01、个人照片及google，因此图片的清晰度较高。在XP操作系统下部分训练或者测试图片无法看清楚，但可用OpenCV正常读取和显示。
 * 正样本来自于`INRIAPerson/96X160H96/Train/pos`总共2416张图片，使用时从中间裁剪出大小为64x128的图片。
 * 负样本来自于`INRIAPerson/Train/neg`有1218张图片，使用时从每一张图片随机裁剪出10张大小为64x128的图片。
 * 将正样本(2416张图片)和负样本(12180张图片)提取HOG特征并投入SVM训练，第一轮训练完成。
 * 拿第一轮训练得到的SVM分类器对原来的负样本(1218张图片)进行检测，将检测出来的行人(误报)保存为HardExample。
 * 将正样本(2416张图片)和负样本(12180张图片)，在加上HardExample一起，提取HOG特征并投入SVM进行第二轮训练。
 * 训练完成，得到最终的SVM分类器。

## 实验结果（Experimental Results）
 * 测试正样本来自于`INRIAPerson/test_64x128_H96/pos`，总共1132张图片（大小为70x134）。
 * 测试负样本来自于`INRIAPerson/test_64x128_H96/neg`（453张图片），每张图片随机裁剪出10张图片（大小为70x134），总共4530张图片。
 * 测试得到结果 TP=795，FN=337，FP=8，TN=4522
  * TP：将行人样本分类为行人的样本数
  * FN：将行人样本分类为非行人的样本数
  * FP：将非行人样本分类为行人的样本数
  * TN：将非行人样本分类为非行人的样本数

  <table border="1" style="text-align: center;">
    <tr>
      <td rowspan="2">分类结果</td>
      <td colspan="2">真实值</td>
    </tr>
      <td>Positive（行人）</td>
      <td>Negative（非行人）</td>
    <tr>
      <td>Positive（行人）</td>
      <td>True Positive (TP)</td>
      <td>False Positive (FP)</td>
    </tr>
    <tr>
      <td>Negative（非行人）</td>
      <td>False Negative (FN)</td>
      <td>True Negative (TN)</td>
    </tr>
  </table>

# 结论（Conclusion）
...
# 参考文献（References）
* Dalal, N. and Triggs, B., “Histograms of oriented gradients for human detection,” in [Computer Vision and
Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on ], 1, 886–893 vol. 1 (June).
* 自己训练SVM分类器进行HOG行人检测 http://blog.csdn.net/masibuaa/article/details/16105073
* 利用Hog特征和SVM分类器进行行人检测 http://blog.csdn.net/carson2005/article/details/7841443
