import os
from graphviz import Digraph

# 方法1：直接设置环境变量（确保路径正确）
os.environ["PATH"] += os.pathsep + r'D:\deep_leaning\Graphviz'

# 方法2：强制指定dot可执行文件路径（终极解决方案）
os.environ["GRAPHVIZ_DOT"] = r'D:\deep_leaning\Graphviz\bin\dot.exe'  # 关键设置！

# 测试生成一个简单图形
try:
    dot = Digraph(comment='Test')
    dot.node('A', 'Input')
    dot.node('B', 'Output')
    dot.edge('A', 'B')
    dot.render('test_graph', format='png', cleanup=True)
    print("图形生成成功！")
except Exception as e:
    print("失败:", e)