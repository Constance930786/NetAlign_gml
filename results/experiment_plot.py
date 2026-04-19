import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_comparative_robustness(csv_path="experiment_results.csv", save_path="experiment_results.png"):
    df = pd.read_csv(csv_path)
    
    df_struct = df[df['attr_noise'] == 0.0].copy()
    
    # 将噪音转化为百分比显示
    df_struct['Noise_Level_Pct'] = df_struct['edge_noise'] * 100
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6), dpi=300)
    
    sns.lineplot(
        data=df_struct, 
        x='Noise_Level_Pct', 
        y='hit5',                 # 展示我们重点强调的 Hit@5 指标
        hue='method',             # 根据不同的算法（Node2Vec vs FINAL）分配不同颜色
        style='method',           # 根据不同算法分配不同线型（实线/虚线）
        markers=True,             # 在数据点上打上标记
        dashes=False, 
        linewidth=2.5,
        markersize=9
    )
    
    plt.title('Hit@5 Robustness under Structural Noise', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Structural Noise Level (Random Edge Removal %)', fontsize=14)
    plt.ylabel('Hit@5 Score', fontsize=14)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0.0, 1.05)
    
    plt.legend(title='Algorithm', loc='lower left', fontsize=12, title_fontsize=13, frameon=True)
    
    # 保存输出
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Figure has been generated successfully and saved into {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_comparative_robustness()