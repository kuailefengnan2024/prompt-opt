# 【功能描述】SkillOpt WebUI 的 ``python -m skillopt_webui`` 入口。
# 【输入】无（由 app.main 解析 --port 等参数）。
# 【输出】启动 Gradio Web 服务。
from skillopt_webui.app import main
main()
