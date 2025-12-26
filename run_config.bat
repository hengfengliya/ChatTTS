@echo off
REM 作用：关闭命令回显（原理：减少终端噪声）
chcp 936 >nul
REM 作用：切换到 GBK 编码（概念：中文控制台兼容）
setlocal enabledelayedexpansion
REM 作用：启用延迟变量扩展（概念：支持动态变量）

REM 作用：设置默认文本（概念：默认值）
set "TEXT=这里是知末平台，1个专业的设计师需求平台"
REM 作用：设置默认文本（概念：固定文案）

REM 作用：设置默认配置（概念：默认参数）
set "SOURCE=local"
REM 作用：默认模型来源（概念：本地模型）
set "FORMAT=mp3"
REM 作用：默认输出格式（概念：音频编码）
set "OUTPUT=output_audio_0.mp3"
REM 作用：默认输出文件名（概念：文件路径）
set "DEVICE=auto"
REM 作用：默认设备（概念：自动选择）
set "SPK_AUDIO="
REM 作用：默认参考音频为空（概念：随机音色）
set "SPK_FILE="
REM 作用：默认音色文件为空（概念：随机音色）

REM 作用：是否修改配置（概念：交互确认）
set "EDIT="
REM 作用：初始化编辑选项（概念：空值起步）
set /p EDIT=是否修改配置？(Y/N，默认N)：
REM 作用：读取用户选择（原理：从标准输入读取）
if /I "%EDIT%"=="Y" goto EDIT_CONFIG
REM 作用：进入配置编辑（概念：条件跳转）
if /I "%EDIT%"=="YES" goto EDIT_CONFIG
REM 作用：进入配置编辑（概念：条件跳转）
goto RUN_DEFAULT
REM 作用：使用默认配置运行（概念：流程跳转）

:EDIT_CONFIG
REM 作用：提示输入合成文本（概念：用户输入）
set /p TEXT=请输入合成文本(默认已内置)：
REM 作用：读取文本输入（原理：从标准输入读取）
if "%TEXT%"=="" set "TEXT=这里是知末平台，1个专业的设计师需求平台"
REM 作用：空输入兜底（概念：默认值）

REM 作用：选择模型来源（概念：模型加载方式）
set /p SOURCE=模型来源(local/huggingface/custom，默认local)：
REM 作用：读取来源输入（原理：从标准输入读取）
if "%SOURCE%"=="" set "SOURCE=local"
REM 作用：空输入兜底（概念：默认值）

REM 作用：选择输出格式（概念：音频编码）
set /p FORMAT=输出格式(mp3/wav，默认mp3)：
REM 作用：读取格式输入（原理：从标准输入读取）
if "%FORMAT%"=="" set "FORMAT=mp3"
REM 作用：空输入兜底（概念：默认值）

REM 作用：设置输出文件名（概念：文件路径）
set /p OUTPUT=输出文件名(默认output_audio_0.%FORMAT%)：
REM 作用：读取输出名（原理：从标准输入读取）
if "%OUTPUT%"=="" set "OUTPUT=output_audio_0.%FORMAT%"
REM 作用：空输入兜底（概念：默认值）

REM 作用：选择设备（概念：CPU/GPU）
set /p DEVICE=设备(auto/cpu/cuda，默认auto)：
REM 作用：读取设备输入（原理：从标准输入读取）
if "%DEVICE%"=="" set "DEVICE=auto"
REM 作用：空输入兜底（概念：默认值）

REM 作用：可选参考音频（概念：零样本音色模拟）
set /p SPK_AUDIO=参考音频路径(可空)：
REM 作用：读取参考音频路径（原理：从标准输入读取）

REM 作用：可选音色嵌入文件（概念：音色特征复用）
set /p SPK_FILE=音色嵌入文件路径(可空)：
REM 作用：读取音色文件路径（原理：从标准输入读取）

:RUN_DEFAULT
REM 作用：准备命令行参数（概念：拼接参数）
set "EXTRA_ARGS="
REM 作用：初始化参数变量（概念：空值起步）
if not "%SPK_AUDIO%"=="" set "EXTRA_ARGS=!EXTRA_ARGS! --spk_audio \"%SPK_AUDIO%\""
REM 作用：加入参考音频参数（概念：条件追加）
if not "%SPK_FILE%"=="" set "EXTRA_ARGS=!EXTRA_ARGS! --spk_file \"%SPK_FILE%\""
REM 作用：加入音色文件参数（概念：条件追加）

REM 作用：设置 Python 输出编码（概念：避免中文乱码）
set "PYTHONIOENCODING=utf-8"
REM 作用：设置环境变量（原理：影响 Python 标准输出）

REM 作用：设置 Python 解释器路径（概念：虚拟环境）
set "PYTHON_EXE=.\.venv\Scripts\python.exe"
REM 作用：指向虚拟环境 Python（概念：隔离依赖）

REM 作用：检查虚拟环境是否存在（概念：运行前校验）
if not exist "%PYTHON_EXE%" goto VENV_MISSING
REM 作用：若不存在则跳转（概念：条件跳转）

REM 作用：执行可配置脚本（概念：命令行运行）
goto RUN_PY
REM 作用：跳转到运行区（概念：流程控制）

:VENV_MISSING
REM 作用：提示虚拟环境缺失（概念：错误提示）
echo 未找到虚拟环境，请先执行部署步骤创建 .venv
REM 作用：输出提示信息（原理：打印到终端）
pause
REM 作用：暂停窗口（原理：等待用户按键）
exit /b 1
REM 作用：返回错误码退出（概念：非零退出）

:RUN_PY
REM 作用：执行可配置脚本（概念：命令行运行）
"%PYTHON_EXE%" "examples\cmd\run_config.py" "%TEXT%" --source "%SOURCE%" --format "%FORMAT%" --output "%OUTPUT%" --device "%DEVICE%" %EXTRA_ARGS%
REM 作用：开始推理（原理：文本转语音）

REM 作用：保持窗口可见（概念：查看运行结果）
pause
REM 作用：暂停窗口（原理：等待用户按键）