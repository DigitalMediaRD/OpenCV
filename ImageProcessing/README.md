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

### 4.2.3 Perspective

函数表达式

    cv2.warpPerspective(input,M,dsize[,dst[,flags[,borderMode[,borderValue]]]])




## 4.3 ImageBlur
图像模糊也称为平滑处理，将一点的像素值调整为与周边像素点相近的值，消除噪声和边缘

### 4.3.1 Meanfiltering

以当前像素点为中心，周边N*N个像素点的平均值替代当前点的像素值


```cv2.blur```函数表达式

    cv2.blur(input,ksize [,anchor [,borderType]])

- ksize为卷积核大小，也就是周边N*N个像素点的区域
    - ***与图像大小的关系***
- anchor为锚点，默认值为(-1，-1)，表示锚点位于卷积核中心
- borderType为边界处理方式

### 4.3.2 GaussianFilter

按像素点与中心点的不同，赋予像素点不同的权重值。越靠近中心点权重越大，反之越小；根据权重值计算邻域内所有像素点的和，作为中心点的值

```cv2.GaussianBlur```函数表达式

    cv2.GaussianBlur(input,ksize, sigmaX [,sigmaY [,borderType]])

- sigmaX为水平方向上的权重值
- sigmaY为垂直方向上的权重值
- 若sigmaX与sigmaY均为0，则默认公式如下
    
    sigmaX=0.3*((width-1)*0.5-1)+0.8
    sigmaY=0.3*((height-1)*0.5-1)+0.8

### 4.3.3 BoxFilter
基于Meanfiltering均值滤波，可选择是否对滤波结果归一化
- 归一化则滤波结果为邻域内点的像素值总和的平均值
- 未归一化则滤波结果为像素值总和

```cv2.boxFilter```函数表达式

    cv2.boxFilter(input,ddepth,ksize [,anchor[,normalize [,borderType]]])

- ddepth为目标[图像深度](https://baike.baidu.com/item/%E5%9B%BE%E5%83%8F%E6%B7%B1%E5%BA%A6/7293591)，一般使用-1表示与原图像的深度一致
- normalize为True时执行归一化操作

### 4.3.4 MedianFilter

将领域内所有像素值排序，取中间值为领域中心点的像素值

```cv2.medianBlur```函数表达式

    cv2.medianBlur(input,ksize)

- ksize若设置为大于1的奇数，则代表为正方形大小的卷积核；否则需把ksize设置为包含行列数目的元组格式

### 4.3.5 BilateralFilter

若像素点与当前点色差较小，则赋予较大权重值；反之赋予较小权重值

```cv2.bilateralFilter```函数表达式

    cv2.bilateralFilter(input,d,sigmaColor,sigmaSpace[,borderType])

- d表示以当前点为中心的邻域直径
- sigmaColor为双边滤波选择的色差范围
- sigmaSpace为空间坐标中的sigma值，值越大表示越多像素点参与滤波计算。当d>0时忽略sigmaSpace，由d决定邻域大小；否则d由sigmaSpace计算得出，与sigmaSpace成比例

### 4.3.6 TwoDimensionConvolution

2D卷积核的自定义函数表达式

    cv2.filter2D(input,ddepth,kernel,anchor[,borderType])

- ddepth为目标图像深度,一般使用-1表示与原图像的深度一致
- kernel为单通道卷积核(一维数组结构)
- anchor图像锚点
- delta修正值，若存在将加上该值作为最终滤波结果
- borderType为边界处理方式

## 4.4 ThresholdProcessing
阈值处理用以剔除图像中像素值高于或低于指定值的像素点

### 4.4.1 GlobalThresholdProcessing
将像素值大于阈值的像素值设置为255，其他像素值设置为0；或大于阈值的像素值设置为0，其他像素值设置为255

全局阈值处理函数表达式

    cv2.threshold(input,thresh,maxval,type):
        return retval, output

- retval为返回的阈值
- thresh为设置的阈值标准
- maxval为阈值类型为THRESH_BINARY和THRESH_BINARY_INV时使用的最大值
- type为阈值类型


### 4.4.2 AdaptiveThresholdProcessing

计算每个像素点邻域的加权平均值来确定阈值，并用该阈值处理当前像素点。适用于色彩明暗差异较大的图像


自适应阈值处理函数表达式

    cv2.adaptiveThreshold(input,maxValue,adaptiveMethod,thresholdType,blockSize,C)
        
- maxValue为最大值
- adaptiveMethod为自适应方法参数，常见的包括
    - cv2.ADAPTIVE_THRESH_MEAN_C:邻域中所有像素点的权重值相同
    - cv2.ADAPTIVE_THRESH_GAUSSIAN_C:邻域中所有像素点的权重值与其到中心点的距离有关，通过高斯方程可计算各个点的权重值
- thresholdType为阈值处理方式
- blockSize为计算局部阈值的邻域的大小
- C为常量，自适应阈值为blockSize指定邻域的加权平均值减去C


## 4.5 MorphologicalTransformations

### 4.5.1 MorphologicalManipulation
会使用一个内核遍历图像，根据内核和图像的位置关系决定内核中心的图像像素点的输出结果。内核可以是自定义的矩阵形式，也可以是调用内置函数返回的形式

自适应阈值处理函数表达式

    cv2.getStructuringElement(shape,ksize)

- shape为内核形状，包括
    - cv2.MORPH_RECT矩形
    - cv2.MORPH_CROSS十字形
    - cv2.MORPH_ELLIPSE椭圆形
- ksize为内核大小


### 4.5.2 Erosion
腐蚀操作遍历图像时，根据内核和图像的位置决定内核中心对应的图像像素点的输出结果。示意图中，0表示背景部分，1表示前景部分；灰色方块表示大小为3x3的矩形内核。腐蚀操作时，依次将内核中心对准每一个单元格，根据内核和前景的位置关系决定当前单元格的值
- 当内核部分或全部处于前景外时，内核中心对应单元格的值设置为0
- 内核完全处于前景内部时，内核中心对应单元格的值才设置为1

腐蚀处理函数表达式

    cv2.erode(input,kernel[,anchor[,iterations[,borderType[,borderValue]]]])

- kernel为内核
- anchor为锚点，默认值为(-1，-1)，表示锚点位于中心
- iterations为腐蚀操作的迭代次数
- borderType为边界处理方式
- borderValue为边界值，由OpenCV自动确定



### 4.5.3 Dilation
与腐蚀操作相反，对图片边界进行扩张

- 当内核完全处于前景外时，内核中心对应单元格的值设置为0
- 内核部分处于前景内部时，内核中心对应单元格的值设置为1

膨胀处理函数表达式

    cv2.dilate(input,kernel[,anchor[,iterations[,borderType[,borderValue]]]])

参数含义与腐蚀相同

### 4.5.4 MorphologyEx
高级形态操作基于腐蚀和膨胀运算，包括开运算、闭运算、形态学梯度运算等

MorphologyEx处理函数表达式

    cv2.morphologyEx(input,op,kernel[,anchor[,iterations[,borderType[,borderValue]]]])

- op为操作类型
    - cv2.MORPH_ERODE执行腐蚀
    - cv2.MORPH_DILATE执行膨胀
参数含义与上述相同

OpenOperation开运算
- 先腐蚀再膨胀

CloseOperation闭运算
- 先膨胀再腐蚀

MorphologicalGradient形态学梯度运算
- 用膨胀操作减去腐蚀操作

BlackHat黑帽运算
- 图像的闭运算减去原图像

TopHat礼貌运算
- 原图像减去开运算结果




# 5 Edges&Contours
## 5.1 EdgeDetection
勾勒出图像灰度值发生急剧变化的部分，检测结果为黑白图像，白色线条表示边缘
### 5.1.1 LaplacianEdgeDetection
使用图像矩阵与拉普拉斯核进行卷积运算，计算图像中任意一点与其在水平方向和垂直方向上4个相邻点平均值的差值

Laplacian函数表达式

    cv2.Laplacian(input,ddepth[,ksize[,scale[,delta[,borderType]]]])

- ddepth为目标图像深度
- ksize用于计算二阶导数滤波器的系数，要求为正奇数
- scale为可选比例因子
- delta为添加到边缘检测结果中的可选增量值
- borderType为边界处理方式

### 5.1.2 SobelEdgeDetection
使用高斯滤波和微分进行卷积运算，结果具有一定的抗噪性

Sobel函数表达式

    cv2.Sobel(input,depth,dx,dy[,ksize[,scale[,delta[,borderType]]]])

- ddepth为目标图像深度
- dx为导数x的阶数
- dy为导数y的阶数
- ksize为扩展的Sobel内核大小，要求为1、3、5、7之一
- scale为可选比例因子
- delta为添加到边缘检测结果中的可选增量值
- borderType为边界处理方式

### 5.1.3 CannyEdgeDetection
比上述检测方法复杂，且上述两种方法可能损失过多的边缘信息或存在很多噪声。Canny检测算法步骤如下

- 使用高斯滤波去除图像噪声
- 使用Sobel核进行滤波，计算梯度
- 在边缘使用非最大值抑制
- 对检测出的边缘使用双阈值以除去假阳性
- 分析边缘之间的连接性，保留真正的边缘，消除不明显的边缘


Canny函数表达式

    cv2.Canny(input,thershold1,thershold2,[,apertureSize[,L2gradient]])

- ddepth为目标图像深度
- thershold1为第一阈值
- thershold2为第二阈值
- apertureSize为计算梯度时使用的Sobel核大小
- L2gradient为标志


## 5.2 ImageContours
图像轮廓指由位于边缘、连续、具有相同颜色和强度的点构成的曲线，可用于形状分析和图像检测识别
### 5.2.1 FindContours
从二值图像中查找轮廓，findContours函数表达式

    cv2.findContours(input,mode,method[,offset])
        return contours,hierarchy


- contours为返回的轮廓
- hierarchy为返回的轮廓层次结构
- mode为轮廓检索模式
- method为轮廓的近似方法
- offset为每个轮廓点移动的可选偏移量

### 5.2.2 DrawContours
drawContours函数表达式

    cv2.drawContours(input,contours,contoursIdx,color[,thickness[,lineType[,hierarchy[,maxLevel[,offset]]]]])

- contours为绘制的轮廓
- contoursIdx为绘制的轮廓的索引，大于等于0时绘制对应的轮廓，负数表示绘制所有轮廓
- hierarchy为返回的轮廓层次结构
- color为BGR格式的颜色结构元组
- thickness控制线条粗细
- lineType控制线型
- hierarchy为返回的轮廓层次结构
- maxLevel为可绘制的最大轮廓层次深度
- offset控制轮廓偏移位置

### 5.2.3 ContoursFeature
轮廓的矩包含了轮廓的各种几何特征，例如面积、位置、角度等

moments函数表达式返回轮廓的矩

    cv2.moments(array[,binaryImage])
        return ret

- ret为轮廓的矩，类型为字典
- array表示轮廓的数组
- binaryImage为True时，会将array对象中所有的非0值设置为1    

contourArea函数表达式返回轮廓的面积

    cv2.contourArea(contour[,oriented])
        return area

- area为轮廓的面积
- contour表示轮廓
- oriented为True时，返回值的正与负表示轮廓是顺时针还是逆时针；为False时返回值为绝对值


arcLength函数表达式返回轮廓的长度

    cv2.arcLength(contour,closed)
        return length

- length为长度
- contour表示轮廓
- closed为True时，表示轮廓封闭

approxPolyDP函数表达式返回轮廓的近似多边形

    cv2.approxPolyDP(contour,epsilon,closed)
        return ret

- ret为近似多边形
- contour表示轮廓
- epsilon为精度，表示近似多边形接近轮廓的最大距离
- closed为True时，表示轮廓封闭



conveHull函数表达式返回轮廓的近似多边形

    cv2.conveHull(contour[,clockwise[,returnPoints]])
        return hull

- hull为返回的凸包，```numpy.ndarry```对象
- contour表示轮廓
- clockwise为True时凸包顺时针，为False时逆时针
- returnPoints为True返回hull包含的凸包关键点坐标，False时返回凸包关键件在轮廓的索引



boundingRect函数表达式返回可包含轮廓大小的矩形

    cv2.boundingRect(contour)
        return rectangle

- rectangle返回直边界矩形，四元组类型
    - 左上角x坐标、左上角y坐标、宽度、高度
- contour表示轮廓


minAreaRect函数表达式返回可包含轮廓大小的最小矩形

    cv2.minAreaRect(contour)
        return box

- box返回旋转矩形，三元组类型
    - 矩阵中心点x坐标、矩阵中心点y坐标，宽度、高度，旋转角度
- contour表示轮廓

boxPoints函数表达式实现矩形的绘制

    cv2.boxPoints(box)
        return points

- points为返回的矩形顶点坐标，浮点型



## 5.3 HoughTransform
### 5.3.1 HoughLines
检测图像中的直线，表达式如下

    cv2.HoughLines(image,rho,theta,threshold)
        return lines

- lines为返回的直线图案
- image要求图像类型为8位单通道二值图像，需要在霍夫变换前先进行处理
- rho为距离精度，默认为1，像素单位
- theta为角度精度，通常为π/180°
- threshold为阈值，越小检测出的直线越多

利用概率活肤变换检测图像中的直线，表达式如下

    cv2.HoughLinesP(image,rho,theta,threshold[,minLineLength[,maxLineGap]])
        return lines

- minLineLength为可接受的直线最小长度，默认0
- maxLineGap位共线线段之间的最大间隔，默认为0

### 5.3.2 HoughCircles
检测图像中的圆，表达式如下

    cv2.HoughCircles(image,method,dp,minDist[,param1[,param2[,minRadius[,maxRadius]]]])
        return circles

- circles为返回的圆图案
- image要求图像类型为8位单通道二值图像，需要在霍夫变换前先进行处理
- method为查找方法
- dp为累加器分辨率，值与图像分辨率成反比。取1时相同，取2时累加器的宽高为输入图像的一半
- minDist为圆心间的最小距离
- param1对应Canny边缘检测的高阈值，默认100
- param2为圆心位置必须到达的投票数，值越大检测出的圆越少，默认100
- minRadius为最小圆半径，低于最小值的圆不会被检测到
- maxRadius为最大圆半径，大于最大值的圆不会被检测到

# 6 Histogram
## 6.1 Foundation
直方图统计图像内各个灰度级出现的次数，横轴表示图像灰度级别，纵轴表示像素灰度级的数量

- RANGE 
- BINS：灰度级分组，将256个灰度量级划分为若干个数量相等的组
- DIMS

### 6.1.1 Hist

绘制函数如下

    matplotlib.pylot.hist(src,bins)

- src为图像数据，要求为一维数组
    - 通过ravel()将三维数组转换为一维
- bins为分组数量

### 6.1.2 CalcHist
OpenCV的直方图查找表达式如下

    cv2.calcHist(image,channels,mask,histSize,ranges)
        return hist


- hist为返回的直方图，大小256的一维数组形式，保存了各个灰度级的数量
- image为原图像，输入形式为[image]
- channels为图像通道，灰度图像为[0]，BGR图像则包含三个通道
- mask为掩模图像，为None时统计整个图像
- histSize()为BINS的值，实际参数形式例如[256]
- ranges为像素值范围，8位灰度图像为[0,255]

### 6.1.3 Histogram
NumPy的直方图查找表达式如下

    np.histogram(image,bins,ranges)
        return hist,edges


- hist为返回的直方图，大小256的一维数组形式，保存了各个灰度级的数量
- bins为灰度级分组数量
- range为像素值范围
- edges为返回的灰度级分组数量边界值



## 6.2 HistogramEqualization
### 6.2.1 NormalHistogramEqualization
将原图像的灰度级均匀映射到全部灰度级范围内，OpenCV的均衡化表达式如下

    cv2.equalization(src)
        return img

- img为均衡化后的图像

### 6.2.2 ContrastLimitedAdaptiveHistogramEqualization
限制对比度自适应直方图均衡化，OpenCV的均衡化表达式如下

    cv2.createCLAHE([clipLimit[,tileGridSize]])
        return retval

- retval为返回的CLAHE对象
- clipLimit为对比度受限的阈值，默认40.0
- tileGridSize为直方图均衡化的网格大小，默认值为(8,8)


## 6.3 TwoDimentionHistogram
二维直方图统计像素的色相和饱和度
### 6.3.1 OpenCV_TwoDimentionHistogram
OpenCV的二维直方图查找表达式如下

    cv2.calcHist(image,channels,histSize,ranges)
        return hist

- hist为返回的直方图，可直接使用```cv2.imshow()```显示
- image为原图像，需要从BGR色彩空间转换为HSV色彩空间，输入形式为[image]
- channels为图像通道，参数为[0,1]时，同时处理色相和饱和度
- histSiz设置BINS的值为[180，256]代表色相为180，饱和度为256
- ranges设置为[0,180,0,255]表示色相取值[0,180]，饱和度取值[0,256]


### 6.3.2 NumPy_TwoDimentionHistogram

NumPy的二维直方图查找表达式如下

    np.histogram2d(x,y,bins,ranges)
        return hist,xedges,yedges


- hist为返回的直方图，大小256的一维数组形式，保存了各个灰度级的数量
- x和y为原图对应通道转换成的一维数组
- bins为灰度级分组数量，例如[180,256]
- range为像素值范围，例如[[0,180],[0,256]]
- xedges为返回的x的直方图的bins边界值
- yedges为返回的y的直方图的bins边界值




# 7 TemplateMatching&Split
## 7.1 TemplateMatching
模板图像在输入图像中滑动，遍历所有像素进行比较，找出和模板最匹配的部分
### 7.1.1 SingleTemplateMatching
只存在一个可能匹配的结果

OpenCV的匹配表达式如下

    cv2.matchTemplate(image,templ,method)
        return result

- image要求为8位或32位浮点类型
- templ为模板图像，数据类型要求与image相同并小于image
- method为匹配方法
    - cv2.TM_SQDIFF_NORMED:归一化方差匹配
    - cv2.TM_SQDIFF:以方差结果为依据进行匹配，完全匹配时为0，否则为一个极大值
    - cv2.TM_CCORR_NORMED:归一化相关匹配
    - cv2.TM_CCORR:相关匹配，输入图像与模板图像相乘，成绩越大匹配度越高，为0时匹配度最低
    - cv2.TM_CCOEFF_NORMED:归一化相关系数匹配
    - cv2.TM_CCOEFF:相关系数匹配，输入图像与其均值的相关值和模板图像与其均值的相关值匹配，为1表示完美匹配，-1为糟糕，0为无相关性
- result为返回结果，它是一个numpy.ndarry对象。若输入图像的大小为W*H,模板图像大小为w * h,则result的大小为( W- w+1)x(H-h+1)，其中的每个值都表示对应位置的匹配结果。当匹配方法为cv2.TM_ SQDIFF 或cv2.TM_ SQDIFF_ NORMED时，匹配结果值越小说明匹配度越高，反之则说明匹配度越低。当匹配方法为cv2.TM_CCORR、Cv2.TM_CCORR_NORMED、cv2.TM_CCOEFF或cv2.TM_CCOEFF_NORMED时，匹配结果值越小说明匹配度越低，反之则说明匹配度越高。

OpenCV处理匹配结果的表达式如下

    cv2.minMaxLoc(src)
        return minVal,maxVal,minLoc,maxLoc

- src为```cv2.matchTemplate```的返回结果
- minVal为src中的最小值，不存在时为NULL
- maxVal为src中的最大值，不存在时为NULL
- minLoc为src中的最小值的位置，不存在时为NULL
- maxLoc为src中的最大值的位置，不存在时为NULL



### 7.1.2 MultiTemplateMatching
匹配多个符合条件的结果，设置阈值，大于等于则代表符合条件

## 7.2 ImageSplit
## 7.2.1 DistanceTransform
分水岭算法：将任意的灰度图像视为地形图表面，其中灰度值高的部分表示山峰和丘陵,而灰度值低的部分表示山谷。用不同颜色的水(标签)填充每个独立的山谷(局部最小值);随着水平面的上升，来自不同山谷(具有不同颜色)的水将开始合并。为了避免出现这种情况，需要在水的汇合位置建造水坝;持续填充水和建造水坝，直到所有山峰和丘陵都在水下。整个过程中建造的水坝将作为图像分割的依据。使用分水岭算法执行图像分割操作时通常包含下列步骤：
    - 将原图像转换为灰度图像
    - 应用形态变换中的开运算和膨胀操作，去除图像噪声，获得图像边缘信息，确定图像背景
    - 进行距离转换，再进行阈值处理，确定图像前景
    - 确定图像的未知区域(用图像的背景减去前景的利余部分)
    - 标记背景图像
    - 执行分水岭算法分割图像

distanceTransform()计算非0值像素到0值像素的距离，函数表达式如下

    cv2.distanceTransform(src,distanceType,maskSize[,dstType])
        return dst

- src为8位单通道的二值图像
- distanceType为距离类型
- maskSize为掩膜大小，可为0、3或5
- dstType为返回的图像类型，默认CV_32F


connectedComponents()将图像中的背景标记为0，将其他图像标记为从1开始的整数，函数表达式如下

    cv2.connectedComponents(src[,connectivity[,ltype]])
        return ret,labels

- labels为返回的编辑结果图像，和src大小相同
- src为要标记的8位单通道图像
- connectivity为4或8，表示连接性
- ltype为返回的标记结果图像类型


watershed()执行分水岭算法分割图像，函数表达式如下

    cv2.watershed(src,markers)
        return ret

- ret为返回的8位或32位单通道图像
- src为输入的8位3通道图像
- markers为输入的32位单通道图像

## 7.2.2 PyrDown
图像金字塔从分辨率的角度分析处理图像。图像金字塔的底部为原始图像，对原始图像进行行梯次向下采样，得到金字塔的其他各层图像。层次越高，分辨率越低，图像越小。通常，每向上一层，图像的宽度和高度就为下一层的一半。常见的图像金字塔可分为高斯金字塔和拉普拉斯金字塔

高斯金字塔有向下和向上两种采样方式。向下采样时，原始图像为第0层，第1次向下采样的结果为第1层，第2次向下采样的结果为第2层，依此类推。每次采样图像的高度和宽度都减小为原来的一半，所有的图层构成高斯金字塔。向上采样的过程和向下采样相反，每次采样图像的高度和宽度都扩大为原来的二倍

pyrDown()执行高斯金字塔向下采样，函数表达式如下

    cv2.pyrDown(src[,dstsize[,borderType]])
        return ret

- ret为返回图像,类型与输入图像相同
- src为输入图像
- dstsize为输出图像大小
- borderType为边界值类型


pyrUp()执行高斯金字塔向上采样，函数表达式如下

    cv2.pyrUp(src[,dstsize[,borderType]])
        return ret

- ret为返回图像,类型与输入图像相同
- src为输入图像
- dstsize为输出图像大小
- borderType为边界值类型

拉普拉斯金字塔的第n层是该高斯金字塔图像减去第n+1层向上采样结果获得的图像

## 7.3 lnteractiveForegroundExiraction
交式前最提取的基本原理为：
- 首先，用一个矩形指定要提取的前景所在的大致范围，然后执行前景提取算法，得到初步结果。初步结果中包含的前景可能并不理想，存在前景未提取完整或者背景被处理为前景等问题。此时需要人工干预(体现交互),用户需要复制原图像作为掩模图像，在其中用白色标注要提取的前景区域，用黑色标注背景区域，标注并不需要很精确。然后，使用掩模图像执行前景提取算法，从而获得理想结果

grabCut()实现前景提取，函数表达式如下

    cv2.grabCut(src,mask1,rect,bgdModel,fgdModel,iterCount[,mode])
        return mask2,bgdModel,fgdModel

- src为输入的8位3通道图像
- mask1为输入的8位单通道掩膜图像，指定图像的哪些区域可能是背景或前景
- mask2为输出的掩膜图像，0表示确定的背景，1表示确定的前景，2表示可能的背景，3表示可能的前景
- bgdModel和fgdModel用于内部计算的临时数组，需定义为大小是1*65的np.float64类型数组，元素均为0
- rect为矩形坐标，要提取的前景图像在矩形内部，将矩形外部视为背景。，mode参数设置为使用矩形模板时，rect参数有效
- iterCount迭代次数
- mode为前景提取模式，包括
    - cv2.GC_INIT_WITH_RECT
    - cv2.GC_INIT_WITH_MASK
    - cv2.GC_EVAL修复模式
    - cv2.GC_EVAL_FREEZE_MODEL固定模式



# 8 FeatureDetection
## 8.1 AngleDetection
图像中的角检测
### 8.1.1 HarrisAngleDetection
cornerHarris()实现图像角检测，函数表达式如下

    cv2.cornerHarris(src,blockSize,ksize,k)
        return dst

- dst为返回结果,numpy.ndarray对象，大小和src相同，每个数组元素对应一个像素点，值越大代表为角的概率越高
- src为8位单通道或浮点值图像
- blockSize为邻域大小，值越大代表检测出的角占区域越大
- ksize为哈里斯角检测器使用的Sobel算子的中孔参数
- k为哈里斯角检测器的自由参数，ksize和k影响检测的灵敏度，值越小检测的角越多，但准确度降低

### 8.1.2 ModifyHarrisAngleDetection
对哈里斯角检测的优化，以便找出更准确的角的位置

cornerSubPix()实现图像角检测的优化，函数表达式如下

    cv2.cornerSubPix(src,corner,winSize,zeroZone,criteria)
        return dst

- dst为返回结果,存储优化后的角信息
- src为8位单通道或浮点值图像
- corner为哈里斯角的质心坐标
- winSize为搜索窗口边长的一半
- zeroZone为零值边长的一半
- criteria为优化查找的终止条件

### 8.1.3 Shi-TomasiAngleDetection
对哈里斯角检测的优化

goodFeaturesToTrack()使用Shi-Tomasi查找图像中的N个最强角，函数表达式如下

    cv2.goodFeaturesToTrack(src,maxCorner,qualityLevel,minDistance)
        return dst

- dst为返回结果,存储检测到的角在原图像中的坐标
- src为8位单通道或浮点值图像
- maxCorner为返回的角的最大数量
- qualityLevel为可接受的角的最低质量
- minDistance为返回的角之间的最小欧几里得距离


## 8.2 FeaturesPointDetection
### 8.2.1 FASTFeaturesDetection
根据像素周围16个像素的强度和阈值等参数判断像素是否为关键点，```cv2.FastFeatureDetector_create()```创建一个FAST对象，然后调用FAST对象的```detect()```执行关键点检测，该方法返回一个关键点列表。每个关键点对象均包含了关键点的角度、坐标、响应强度和邻域大小等信息

### 8.2.2 SIFTFeaturesDetection
图像中的角具有旋转不变特征，但放大时可能产生变化。SIFT用于查找图像中初度不变特征，返回图像中的关键点。```cv2.SIFT_create()```创建一个SIFT对象，然后调用SIFT对象的```detect()```执行关键点检测

### 8.2.3 ORBFeaturesDetection
基础上再进行改进，```cv2.ORB_create()```创建一个ORB对象，然后调用ORB对象的```detect()```执行关键点检测