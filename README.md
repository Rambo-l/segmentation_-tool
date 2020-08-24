#### 语义分割训练前处理工具
#### 主要功能：
#####          FUNC1：视频素材转换成图像
#####          FUNC2：把标注后json文件和对应的jpg文件从混合的文件夹中提取出来
#####          FUNC3：Json_to_Dataset功能.从Json文件获得标签文件
#####          FUNC4：Get_JPG_PNG从上一步的Dataset文件中提取训练图像和训练标签
#####          FUNC5：从训练集中随机选取一定比例的图像和标签作为验证集图像和标签
#####          FUNC6：由模型输出标签和人工标签计算得到MIOU和MPA
#### BY LiangBo