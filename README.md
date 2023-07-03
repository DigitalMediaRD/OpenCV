# 基础语法
- [AUTOMATE THE BORING STUFF WITH PYTHON](https://automatetheboringstuff.com/2e/chapter0/)，学习没有捷径，建议尝试阅读英文原版；若实在吃力可阅读[中文译版](https://kdocs.cn/l/cvONqPjqwC5d)

# 环境配置

## Conda

其特点在于可创建多个不同版本的python环境实现互相独立的多环境管理；缺点在于多个虚拟环境的创建将占用大量硬盘空间，并且当涉及其他需要额外进行编译操作的程序时，操作处理起来略微繁琐.安装后在开始菜单找到```Anaconda/Minconda```的文件夹，并点击文件夹内的```Spyder```图标启动编译器或```Anaconda Prompt (Miniconda3)```图标打开命令行交互窗口


- [Anaconda](https://www.anaconda.com/)/[Minconda](https://docs.conda.io/en/latest/miniconda.html)，作用类似，前者相比后者集成了其他开发工具和可视化界面的操作；后者仅包括精简的命令行窗口功能

	- [英文文档](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html)/[中文文档](https://anaconda.org.cn/anaconda/user-guide/getting-started/)

- [Python资源库在线下载命令](https://www.runoob.com/w3cnote/python-pip-install-usage.html)

### 其他设置

- 在线资源库更换为国内资源：打开 ```Conda窗口界面``` ，永久更换在线资源链接输入下列语句之一

        # 清华源
        pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
        # 阿里源
        pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
        # 腾讯源
        pip config set global.index-url http://mirrors.cloud.tencent.com/pypi/simple
        # 豆瓣源
        pip config set global.index-url http://pypi.douban.com/simple/# 

- 临时更换在线资源链接输入下列语句之一

        # 清华源
        pip install 本次下载的资源包名称 -i https://pypi.tuna.tsinghua.edu.cn/simple
        # 阿里源
        pip install 本次下载的资源包名称 -i https://mirrors.aliyun.com/pypi/simple/
        # 腾讯源
        pip install 本次下载的资源包名称 -i http://mirrors.cloud.tencent.com/pypi/simple
        # 豆瓣源
        pip install 本次下载的资源包名称 -i http://pypi.douban.com/simple


### 暂无需要接触创建虚拟环境可跳过此小节
Conda移植虚拟环境到其他设备的[操作方法](https://blog.csdn.net/buweifeng/article/details/124733123?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-124733123-blog-115385868.t0_layer_searchtargeting_sa&spm=1001.2101.3001.4242.2&utm_relevant_index=3)

	#安装打包资源库
	conda install -c conda-forge conda-pack
	#当前虚拟环境导出包
	conda pack -n env_name
	#登陆需要安装环境的机器
	cd yourpath
	# 解压
	tar zxf target_file.tar.gz
	# 激活环境
	conda activate /yourpath/bin/activate 
	# 查看python的路径
	which python





## OpenCV配置

- pip安装OpenCV

        pip install opencv-python

- [OpenCV官方安装包](https://opencv.org/releases/)，解压后将```\build\python\cv2```内的```cv.pyd```文件复制到python安装路劲下的```\Lib\site-packages\cv2```中
        在命令行窗口界面使用pip安装opencv_contrib_python
        若执行import cv2 命令无报错，则说明安装成功


