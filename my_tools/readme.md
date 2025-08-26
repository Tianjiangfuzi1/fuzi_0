1. 运行脚本

在命令行中使用以下格式运行脚本：

python search_code.py <目录路径> <搜索内容> [选项]

3. 参数说明

    目录路径: 要搜索的根目录路径

    搜索内容: 要搜索的文本或模式

    -t 或 --type: 搜索类型，可选值:

        text: 普通文本搜索 (默认)

        class: 类定义

        function: 函数定义

        import: 导入语句

    -o 或 --output: 输出文件路径 (可选，默认为指定搜索的文件夹下面生成的 search_results.txt)

    -c 或 --case-sensitive: 区分大小写 (默认不区分)

4. 示例用法


        搜索当前目录及其子目录中包含 LaneLoss 文本的所有Python文件：

python search_code.py . LaneLoss

        搜索特定目录中所有包含 build_optimizer 函数定义的Python文件：

python search_code.py /path/to/your/project build_optimizer -t function

        搜索包含 mmcv 导入语句的Python文件，区分大小写：

python search_code.py ./mmseg mmcv -t import -c

        搜索包含 nms_thres 文本的Python文件，并将结果保存到指定文件：

python search_code.py . nms_thres -o nms_results.txt

        搜索包含 Anchor3DLane 类定义的Python文件：

python search_code.py . Anchor3DLane -t class