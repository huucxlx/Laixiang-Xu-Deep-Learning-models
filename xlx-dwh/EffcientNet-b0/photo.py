import graphviz

dot = graphviz.Digraph(comment='Model Structure', node_attr={'fontname': 'Helvetica,Arial,sans-serif'},
                       edge_attr={'fontname': 'Helvetica,Arial,sans-serif'})
dot.attr(rankdir='TB', size='8,8', dpi='300')

# 输入层
dot.node('Input', '224×224×3', shape='box', style='filled', fillcolor='lightblue')

# Stem Conv
dot.node('Stem', 'Stem Conv (k=3, s=2, p=1) → 32ch\nBN + SiLU', shape='box', style='filled', fillcolor='lightgreen')

# Stage1
with dot.subgraph(name='cluster_Stage1') as stage1:
    stage1.attr(label='Stage1 (x1)', style='filled', fillcolor='lightyellow')
    stage1.node('IR1',
                'InvertedResidual (k=3, s=1, p=1) → 16ch\nExpand Conv(1×1) → BN + SiLU\nDWConv3×3 → BN + SiLU\nProject Conv(1×1) → BN',
                shape='box')

# Stage2
with dot.subgraph(name='cluster_Stage2') as stage2:
    stage2.attr(label='Stage2 (x2)', style='filled', fillcolor='lightyellow')
    for i in range(2):
        stage2.node(f'IR2_{i}',
                    f'InvertedResidual (k=3, s=2, p=1) → 24ch\nExpand Conv(1×1) → BN + SiLU\nDWConv3×3 → BN + SiLU\nProject Conv(1×1) → BN',
                    shape='box')
        if i > 0:
            stage2.edge(f'IR2_{i - 1}', f'IR2_{i}')

# Stage3
with dot.subgraph(name='cluster_Stage3') as stage3:
    stage3.attr(label='Stage3 (x2)', style='filled', fillcolor='lightyellow')
    for i in range(2):
        stage3.node(f'IR3_{i}',
                    f'InvertedResidual (k=5, s=2, p=2) → 40ch\nExpand Conv(1×1) → BN + SiLU\nDWConv5×5 → BN + SiLU\nProject Conv(1×1) → BN',
                    shape='box')
        if i > 0:
            stage3.edge(f'IR3_{i - 1}', f'IR3_{i}')

# Stage4
with dot.subgraph(name='cluster_Stage4') as stage4:
    stage4.attr(label='Stage4 (x3) → 80ch ★ SKBlock启用', style='filled', fillcolor='lightyellow')
    for i in range(3):
        with stage4.subgraph(name=f'cluster_Branches_{i}') as branches:
            branches.attr(label=f'SKBlock {i + 1}', style='filled', fillcolor='lightpink')
            branches.node(f'Branch1_{i}', 'DWConv3×3 → BN + SiLU', shape='box')
            branches.node(f'Branch2_{i}', 'DWConv5×5 → BN + SiLU', shape='box')
            branches.node(f'Attention_{i}', 'GAP → FC(→C/2) → FC(→M) → Softmax', shape='box')
            branches.edge(f'Branch1_{i}', f'Attention_{i}')
            branches.edge(f'Branch2_{i}', f'Attention_{i}')
        if i > 0:
            stage4.edge(f'Attention_{i - 1}', f'Branch1_{i}')
            stage4.edge(f'Attention_{i - 1}', f'Branch2_{i}')

# Stage5
with dot.subgraph(name='cluster_Stage5') as stage5:
    stage5.attr(label='Stage5 (x3) → 112ch', style='filled', fillcolor='lightyellow')
    for i in range(3):
        with stage5.subgraph(name=f'cluster_SK5_{i}') as sk5:
            sk5.attr(label=f'SKBlock {i + 1} (r=2)', style='filled', fillcolor='lightpink')
            sk5.node(f'SK5_Branch1_{i}', 'DWConv3×3 → BN + SiLU', shape='box')
            sk5.node(f'SK5_Branch2_{i}', 'DWConv5×5 → BN + SiLU', shape='box')
            sk5.node(f'SK5_Attention_{i}', 'GAP → FC(→C/2) → FC(→M) → Softmax', shape='box')
            sk5.edge(f'SK5_Branch1_{i}', f'SK5_Attention_{i}')
            sk5.edge(f'SK5_Branch2_{i}', f'SK5_Attention_{i}')
        if i > 0:
            stage5.edge(f'SK5_Attention_{i - 1}', f'SK5_Branch1_{i}')
            stage5.edge(f'SK5_Attention_{i - 1}', f'SK5_Branch2_{i}')

# Stage6
with dot.subgraph(name='cluster_Stage6') as stage6:
    stage6.attr(label='Stage6 (x4) → 192ch', style='filled', fillcolor='lightyellow')
    for i in range(4):
        with stage6.subgraph(name=f'cluster_SK6_{i}') as sk6:
            sk6.attr(label=f'SKBlock {i + 1} (r=3)', style='filled', fillcolor='lightpink')
            sk6.node(f'SK6_Branch1_{i}', 'DWConv3×3 → BN + SiLU', shape='box')
            sk6.node(f'SK6_Branch2_{i}', 'DWConv5×5 → BN + SiLU', shape='box')
            sk6.node(f'SK6_Attention_{i}', 'GAP → FC(→C/3) → FC(→M) → Softmax', shape='box')
            sk6.edge(f'SK6_Branch1_{i}', f'SK6_Attention_{i}')
            sk6.edge(f'SK6_Branch2_{i}', f'SK6_Attention_{i}')
        if i > 0:
            stage6.edge(f'SK6_Attention_{i - 1}', f'SK6_Branch1_{i}')
            stage6.edge(f'SK6_Attention_{i - 1}', f'SK6_Branch2_{i}')

# Stage7
with dot.subgraph(name='cluster_Stage7') as stage7:
    stage7.attr(label='Stage7 (x1) → 320ch', style='filled', fillcolor='lightyellow')
    with stage7.subgraph(name='cluster_SK7') as sk7:
        sk7.attr(label='SKBlock (r=3)', style='filled', fillcolor='lightpink')
        sk7.node('SK7_Branch1', 'DWConv3×3 → BN + SiLU', shape='box')
        sk7.node('SK7_Branch2', 'DWConv5×5 → BN + SiLU', shape='box')
        sk7.node('SK7_Attention', 'GAP → FC(→C/3) → FC(→M) → Softmax', shape='box')
        sk7.edge('SK7_Branch1', 'SK7_Attention')
        sk7.edge('SK7_Branch2', 'SK7_Attention')

# Top Conv
dot.node('TopConv', 'Top Conv (k=1, s=1, p=0) → 1280ch\nBN + SiLU', shape='box', style='filled', fillcolor='lightgreen')

# GeM Pool
dot.node('GeMPool', 'GeM Pool (p=3.0) → 自适应幂池化', shape='box', style='filled', fillcolor='lightblue')

# Classifier
dot.node('Classifier', 'Dropout(0.1) → FC(1280→n)', shape='box', style='filled', fillcolor='lightgreen')

# 输出层
dot.node('Output', 'Output (n classes)', shape='box', style='filled', fillcolor='lightblue')

# 连接各层
dot.edge('Input', 'Stem')
dot.edge('Stem', 'IR1')
dot.edge('IR1', 'IR2_0')
dot.edge('IR2_1', 'IR3_0')
dot.edge('IR3_1', 'Branch1_0')
dot.edge('IR3_1', 'Branch2_0')
dot.edge('Attention_2', 'SK5_Branch1_0')
dot.edge('Attention_2', 'SK5_Branch2_0')
dot.edge('SK5_Attention_2', 'SK6_Branch1_0')
dot.edge('SK5_Attention_2', 'SK6_Branch2_0')
dot.edge('SK6_Attention_3', 'SK7_Branch1')
dot.edge('SK6_Attention_3', 'SK7_Branch2')
dot.edge('SK7_Attention', 'TopConv')
dot.edge('TopConv', 'GeMPool')
dot.edge('GeMPool', 'Classifier')
dot.edge('Classifier', 'Output')

# 渲染图形
dot.render('complete_model_structure', format='png', cleanup=True, view=True)
