# 图像处理基础
## Numpy
### 数组创建
- array()#将线性表类型的数据结构转化为数组
- zeros()#指定维度的数组，所有元素都为0
- arange()#创建元素值按规则递增的数组
- linspace(a,b,c)
- indices()
- ones()#与zeros()相同，但元素皆为1
### 数组形状
- shape()#查看或更改数组形状
- reshape()#更改并直接返回新数组
- resize()#改变形状更改个数
- ravel()#多维转一维
### 索引切片
与列表操作相似
### 矩阵运算


## ImageOperation
### Read、Write、Show
计算机对图像的操作与对其他文本文件的操作类型一致
- Read
    - 导入相关的程序语言包以便对图像进行处理
    - 调用相关的```Read()```确定需要操作的图像文件，并在程序语言中创建变量以达到图像的信息载入内存后得以保存

- Write
    - 将变量保存的图像数据写入为图像文件格式
    - 调用相关的```Write()```并输入输出路径、格式类型

- Show
    - 在程序语言中实时预览```Read()```载入内存的图像信息

[图像数据的保存](https://blog.csdn.net/u010089444/article/details/52738479)


### VideoRead、Write
计算机对视频的操作逻辑与图像大体一致，由于文件类型不一样导致存在细节上的区别

- Read
    - 导入相关的程序语言包以便对图像进行处理
    - 调用相关的```Read()```确定需要操作的视频文件，并在程序语言中创建变量以达到视频的信息载入内存后得以保存
        - 增加了对帧率、视频尺寸的获取
        - ```Read()```调用后返回两个类型的返回值
            
                returnValue,frame=vc.read()
                returnValue bool类型，若读取当前视频帧成功则返回True，否则False
                frame 保存读取的图片帧数据
    - *由于视频为多张图片排列而成，因此播放视频在结构上可以理解为顺序显示一连串的图片。通过循环调用图像的```imshow()```，顺序显示每次获取到的图片帧数据以达到播放视频的效果*
        - ```cv2.waitKey(int ms)```设置播放视频帧的方法延迟执行


- Write
    - 此处将本地视频通过程序读取，然后调用```Write()```方法写入为新视频文件，因此需要先执行读操作
    - 调用相关的```Write()```并输入输出路径、视频编码标准

            cv2.VideoWriter_fourcc('X','V','I','D')#XVID的MPEG-4编码格式，后缀为.avi

### CameraCapture
待补充

### ProcessingGray
灰度图像的结构：单通道的二维数组，3行3列的结构如下

    img=[
        [128 138 225]
        [127 154 221]
        [84 54 111]
        ]

### ProcessingRGB
彩色图像的结构：3通道的三维数组，3行3列，每个元素包含3个值的结构如下

    img=[
        [[128 138 225]
        [127 154 221]
        [84 54 111]]
        
        [[128 138 225]
        [127 154 221]
        [84 54 111]]

        [[128 138 225]
        [127 154 221]
        [84 54 111]]
        ]


### SeparatingRGB
矩阵拆分方法

```cv2.split()```拆分方法

合并


## ImageCalculation

### 加法运算
图像的矩阵进行相加，选择的相加方法不一样，其结果也存在区别。若像素相加结果大于256
- “+”运算符按256取余数
- ```cv2.add(img1,img2)```取255

### 加权加法
两张图像混合操作但相加不再是简单的直接叠加，而是混合模式下设置两张图片的所占比重不同实现存在主次关系的图像混合，计算公式

    img1*alpha+img2*beta+gamma





# 4 ImageConverting
## 4.1 ColorConverting
色彩空间作为图像在计算机内的存储方式，其存在若干分类。不同种类分别有着擅长解决的问题领域
- RGB
- GRAY
- XYZ
- YCrCb
- HSV

转换函数```cv2.cvtColor()```基本格式为

    output=cv2.cvtColor(input,code[,dstCn])# dstCn表示目标图像的通道数
    # code转换类型如下
    cv2.COLOR_BGR2RGB
    cv2.COLOR_BGR2Gray
    cv2.COLOR_BGR2HSV
    cv2.COLOR_BGR2YCrCb
    cv2.COLOR_RGB2BGR
    cv2.COLOR_RGB2Gray
    cv2.COLOR_RGB2HSV
    cv2.COLOR_RGB2YCrCb


### 4.1.1 RGB
Red、Green、Blue基本颜色代表图像的色彩组成，OpenCV中的色彩空间为BGR，即通道顺序为B、G、R表示图像。

### 4.1.2 GRAY
8位灰度空间，取值范围[0,255]，色彩空间转换的默认计算公式为

    Gray=0.299R+0.587G+0.114B

### 4.1.3 YCrCb
亮度Y、红色Cr、蓝色Cb表示，RGB色彩空间转换的默认计算公式为

    Y=0.299R+0.587G+0.114B
    Cr=0.713(R-Y)+delta
    Cb=0.564(B-Y)+delta

根据图像类型的差异，delta的取值也不同
    
    delta=128 # 8位图像
    delta=32767 # 16位图像
    delta=0.5 # 单精度图像


### 4.1.4 HSV
色调Hue、饱和度Saturation、和亮度Value表示，色调取值范围[0<sup>o</sup>，360<sup>o</sup>]，从红色开始按逆时针方向计算

饱和度S表示颜色接近光谱色的程度，光谱中白光比例越低饱和度越高，颜色越深。取值范围为[0,1]

亮度V取值为[0,1]

RGB色彩空间转换的计算公式

    V=max(R，G，B)
    根据V的取值，S的公式也不同
    if V!=0
        S=(V-min(R，G，B))/V
    else
        S=0

    根据V的取值，H的公式也不同
    if V==R
        H=60(G-B)/(V-min(R，G，B))
    else if V=G
        H=120+ (60(B-R)/(V-min(R，G，B)))
    else if V=B
        H=240+ (60(R-G)/(V-min(R，G，B)))

若H<0，则H=H+360


## 4.2 GeometryConverting
### 4.2.1 Scale
缩放函数表达式

    output=cv2.resize(input,dsize[,dst[,fx[,fy[,interpolation]]]])
    dsize # 转换后的图像大小
    fx # 水平方向的缩放比例
    fy # 垂直方向的缩放比例
    interpolation # 插值方式

根据dsize参数是否为空，也存在不同的参数要求
- 不为None，不管是否设置参数fx、fy，转换后的图片大小都由dsize来确定
- 为None，参数fx和fy不能为0。则转换后的图像宽度为```round(原始宽度*fx)```，高度为```round(原始高度*fy)```

### 4.2.2 Flip

```flipCode```代表翻转类型，为0时绕x轴翻转，大于0时绕y轴翻转；小于0的整数时同时绕x和y翻转





### 4.2.3 Affine
*仿射* 函数表达式

    cv2.warpAffine(input,M,dsize[,dst[,flags[,borderMode[,borderValue]]]])

- M是2x3的转换矩阵，调整矩阵内对应的元素值可实现平移、旋转等多种操作
    $$
    \left[
    \begin{matrix}
        h & 0 & m \\\\
        0 & v & n
    \end{matrix}
    \right] 
    $$
- dsize为转换后的图像大小
- flags、borderMode、borderValue为带默认值的参数，通常可省略


Translation

仿射矩阵M的元素m、n控制平移
- m>0，图像右移，反之左移
- n>0，图像下移，反之上移

Scale

仿射矩阵M的元素h、v控制缩放
- h控制水平方向的缩放
- v控制垂直方向的缩放

Rotation

函数表达式

    cv2.getRotationMatrix2D(center,angle,sacle)

- center控制旋转的中心坐标，组成结构包含两个变量
- angle控制旋转角度，正数逆时针，反之顺时针
- sacle控制缩放



Mapping
函数表达式

    cv2.getAffineTransform(input,output)

将input和output图像中分别取三个点作为平行四边形的左上角、右上角和左下角，按原图和目标图像与3个点的坐标关系计算所有像素的转换矩阵

Perspective

函数表达式

    cv2.warpPerspective(input,M,dsize[,dst[,flags[,borderMode[,borderValue]]]])






