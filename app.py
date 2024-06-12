import os
from logging import getLogger
from apps.grapp import demo
from dotenv import load_dotenv

load_dotenv()
logger = getLogger(__name__)

if __name__ == '__main__':
    # 获取服务器端口
    if (p:=os.getenv('GRADIO_APP_SERVER_PORT')).isdigit():
        server_port = int(p)
    else:
        logger.warning(f"Invalid server port: {p}, using default port 8080")
        server_port = 8080
        
    # 启动Gradio应用
    demo().launch(
        root_path=os.getenv('GRADIO_ROOT_PATH'), 
        server_port=server_port,
        server_name='0.0.0.0',
        # share=True
    )
