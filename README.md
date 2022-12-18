# <p align="center">CyberWaifu</p>
[中文](README.md "中文") [English](eng-README.md "English") [日本語](jp-README.md "日本語")

> ### 这是一个基于TTS+VITS的ChatGPT语音对话程序!

效果演示BiliBIli:[《青春猪头少年不会梦见赛博女友》](https://www.bilibili.com/video/BV1rv4y1Q7eT "BiliBili")

**当前支持功能：**
* [x] ChatGPT的对话聊天
* [x] 回答转语音
* [x] 多角色语音

**预计支持功能：**
* [ ] 语音识别对话 (研发了一款真正人性化的智能语音Q宝
* [ ] 对接Live2D的Web版本
* [ ] 对接Marai机器人

# 目录
* [1.安装环境：](#1.)
	* 1.1 [使用cd命令进入项目文件夹](#cd)
	* 1.2 [创建Python虚拟环境:](#Python:)
	* 1.3 [进入创建好的虚拟环境:](#venv)
	* 1.4 [pip安装项目所需要的库文件:](#pip:)
* [2.导入模型到根目录model文件夹（如果没有自行创建):](#.model)
	* 2.1 [双击导入model](#cd1)
* [3.运行（快和我的老婆们对话吧:](#.:)
	* 3.1 [获取ChatGPT Token](#ChatGPTToken)
	* 3.2 [2.开始和CyberWaifu聊天](#CyberWaifu)
* [4.鸣谢](#CyberWaifuthank)
## <span id="1.">1.安装环境：</span>
> **安装anaconda环境或Python>=3.7**
> 
> **本例使用的环境名称是：chatWaifu**

### <span id="cd">1.1 使用cd命令进入项目文件夹</span>
`cd 你的项目路径`
![](readme/5.png)
### <span id="Python:">1.2 创建Python虚拟环境:</span>

Conda:`conda create --name CyberWaifu python=3.10`
![](readme/1.png)
![](readme/2.png)
Python:`python -m venv chatWaifu`
![](readme/6.png)

### <span id="venv">1.3 进入创建好的虚拟环境:</span>
Conda:`conda activate chatWaifu`
![](readme/3.png)

Python:`.\chatWaifu\Scripts\activate.bat`
![](readme/7.png)

### <span id="pip:">1.4 pip安装项目所需要的库文件:</span>
`pip install -r requirement.txt`
![](readme/4.png)

## <span id=".model">2.导入模型到根目录model文件夹:</span>
Google Drive:https://drive.google.com/file/d/1tMCafhnUoL7FbevVQ44VQi-WznDjt23_ 

阿里云盘: https://www.aliyundrive.com/s/9JEj1mp1ZRv 提取码: m2y3

### <span id="cd1">2.1移动到项目根目录下双击导入model</span>

## <span id=".:">3.运行（快和我的老婆们对话吧:</span>
日语版：`python ChatWaifu1.1.py`

中文版：`python ChatWaifuCN.py`

### <span id="ChatGPTToken">3.1 获取ChatGPT Token</span>
#### 在浏览器登入https://chat.openai.com
#### 按F12进入开发控制台
#### 找到 应用程序 -> cookie -> __Secure-next-auth.session-token
#### 将值复制进入终端并回车

### <span id="CyberWaifu">3.2 开始和CyberWaifu聊天</span>

<video width="640" height="480" controls>
    <source src="readme/example.mp4" type="video/mp4">
</video>


## <span id="CyberWaifuthank">4.鸣谢：</span>
- [MoeGoe_GUI]https://github.com/CjangCjengh/MoeGoe_GUI
- [Pretrained models]https://github.com/CjangCjengh/TTSModels
- [PyChatGPT]https://github.com/terry3041/pyChatGPT
