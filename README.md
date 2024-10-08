# 基础语法
- [AUTOMATE THE BORING STUFF WITH PYTHON](https://automatetheboringstuff.com/2e/chapter0/)，学习没有捷径，建议尝试阅读英文原版；若实在吃力可阅读[中文译版](https://kdocs.cn/l/cvONqPjqwC5d)

# 环境配置

## Conda

其特点在于可创建多个不同版本的python环境实现互相独立的多环境管理；缺点在于多个虚拟环境的创建将占用大量硬盘空间，当涉及需要额外进行编译操作的程序时，处理略繁琐。因此避免安装在系统盘，或者定期清理安装在系统盘的资源包


- [Anaconda](https://www.anaconda.com/)/[Minconda](https://docs.conda.io/en/latest/miniconda.html)，作用类似，前者相比后者集成了其他开发工具和可视化界面的操作；后者仅包括精简的命令行窗口功能

	- [英文文档](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html)/[中文文档](https://anaconda.org.cn/anaconda/user-guide/getting-started/)

安装后在开始菜单找到```Anaconda/Minconda```的文件夹，并点击文件夹内的```Spyder```图标启动编译器或```Anaconda(Miniconda3) Prompt```图标打开命令行交互窗口

- [Python资源库在线下载命令](https://www.runoob.com/w3cnote/python-pip-install-usage.html)

### 创建虚拟环境
Anaconda Prompt创建新的虚拟环境

        #[]内为自定义参数，代表新环境名称，版本号。自定义时注意去除[]
        conda create -n [name] python=[version]

        #查看所有创建的虚拟环境
        conda info -e 

        #激活某个虚拟环境
        conda activate [name]

        #删除某个虚拟环境
        conda remove -n [name] --all

其中```name```的参数表示为在Anaconda中，在默认位置为安装的虚拟环境创建一个同名文件夹，将虚拟环境所有配置文件放置在该同名文件夹下。激活虚拟环境时自动在默认位置根据文件夹名称进行检索

### 组合使用
Pycharm创建工程调用Anaconda创建的虚拟环境流程如下  
创建新工程时选择黄框内代表已存在操作系统中的虚拟环境
![](https://github.com/DigitalMediaRD/OpenCV/blob/main/res/001.png)
在左侧列表中选择对应的选项，下拉菜单中根据实际需要选择关联的编译器
![](https://github.com/DigitalMediaRD/OpenCV/blob/main/res/002.png)


### 导出为可执行程序

导出后缀为```.exe```的可执行程序

        pip install pyinstaller

        pyinstaller -F [path/FileName.py]

若执行打包后的程序出现模块丢失，可能是部分模块安装路径不统一。按照下列命令查询安装路径，并尝试卸载模块重新安装

        python

        import [module]

        print([module].__file__)

### 在线下载

[其他](https://blog.csdn.net/javastart/article/details/102563461)

Anaconda/Miniconda的在线资源安装保存路径默认在子目录下，若需要修改资源保存路径则需要将新目标文件夹的访问权限设置为完全控制，并执行下列命令

        #[]内的路径参数为修改后，创建虚拟环境的安装路径
        conda config --add envs_dirs [E:\Projects\PythonEnvs]

        #[]内的路径参数为修改后，下载资源包的保存路径
        conda config --add pkgs_dirs [E:\Projects\Anaconda\PythonEnvs]

        #查看配置信息，确认是否修改成功
        conda config --show

- 在线资源库更换为国内资源：打开 ```Anaconda(Miniconda3) Prompt``` ，永久更换在线资源链接输入下列语句之一

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


### 虚拟环境移植
Conda移植虚拟环境到同类型操作系统的其他设备[方法](https://blog.csdn.net/buweifeng/article/details/124733123?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-1-124733123-blog-115385868.t0_layer_searchtargeting_sa&spm=1001.2101.3001.4242.2&utm_relevant_index=3)

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
	python --version





## OpenCV配置

- pip安装OpenCV

        pip install opencv-python

- [OpenCV官方安装包](https://opencv.org/releases/)或[压缩包](https://kdocs.cn/l/ciofIJvdWvrU)，解压后将```\build\python\cv2```内的```cv.pyd```文件复制到python安装路径下的```\Lib\site-packages\cv2```中

        在命令行窗口界面使用pip安装opencv_contrib_python
        若执行import cv2 命令无报错，则说明安装成功


