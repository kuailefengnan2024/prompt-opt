# 【功能描述】prompt-opt WebUI 的 ``python -m promptopt_webui`` 入口。
# 【输入】无（由 app.main 解析 --port 等参数）。
# 【输出】启动 Gradio Web 服务。
from promptopt_webui.app import main
main()
