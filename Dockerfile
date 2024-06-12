# 使用精简版本的python作为基础镜像，bullseye代表debian 11, 适合在生产环境中使用
FROM python:3.10-slim-bullseye

RUN sudo apt update
RUN sudo apt install ffmpeg

# 
WORKDIR /code

# 把当前文件夹的所有文件复制到工作文件夹
COPY . /code/

# 检查文件是否存在并报错
# RUN if [ ! -f "config.yml" ]; then echo "请先按照config.example.yml 创建配置文件： 'config.yml'"; exit 1; fi

# 设置时区
RUN rm /etc/localtime
RUN ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# 
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip config set global.extra-index-url "http://mirrors.aliyun.com/pypi/simple/ https://pypi.mirrors.ustc.edu.cn/simple/ http://pypi.hustunique.com/ http://pypi.douban.com/simple/ http://pypi.sdutlinux.org/"
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
# RUN apt-get update && apt-get install -y nano

# 复制字体文件到容器中
COPY static/resources/fonts/STSONG.TTF /usr/local/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/

# 
CMD ["python","main.py"]