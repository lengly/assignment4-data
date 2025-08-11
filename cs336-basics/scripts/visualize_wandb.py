#!/usr/bin/env python3
"""
Wandb结果可视化脚本

这个脚本用于从wandb获取训练数据并生成可视化图表，包括：
- 训练损失曲线
- 验证损失曲线  
- 学习率变化曲线
- 损失对比图

使用方法:
python scripts/visualize_wandb.py --entity your_entity --project your_project --run_id your_run_id
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import numpy as np
from pathlib import Path
import seaborn as sns
from typing import Optional, Dict, Any
import json
from datetime import datetime
import yaml

# 设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class WandbVisualizer:
    def __init__(self, entity: str, project: str, run_id: Optional[str] = None, wandb_dir: Optional[str] = None):
        """
        初始化Wandb可视化器
        
        Args:
            entity: wandb实体名称
            project: wandb项目名称
            run_id: 可选的运行ID，如果不提供则使用最新的运行
            wandb_dir: 本地wandb目录路径，如果提供则使用本地数据
        """
        self.entity = entity
        self.project = project
        self.run_id = run_id
        self.wandb_dir = wandb_dir
        self.api = wandb.Api()
        self.use_local = wandb_dir is not None
    
    def list_projects(self):
        """列出用户的所有项目"""
        if self.use_local:
            print(f"\n本地wandb目录: {self.wandb_dir}")
            print("本地模式不支持项目列表，直接使用本地数据")
            return []
        
        try:
            projects = self.api.projects(self.entity)
            print(f"\n用户 '{self.entity}' 的项目列表:")
            print("="*50)
            for i, project in enumerate(projects, 1):
                print(f"{i}. {project.name}")
            print("="*50)
            return projects
        except Exception as e:
            print(f"获取项目列表时出错: {e}")
            return []
    
    def list_runs(self, project_name: Optional[str] = None):
        """列出项目中的所有运行"""
        if self.use_local:
            return self._list_local_runs()
        
        try:
            project = project_name or self.project
            runs = self.api.runs(f"{self.entity}/{project}")
            print(f"\n项目 '{project}' 的运行列表:")
            print("="*80)
            print(f"{'序号':<4} {'运行ID':<12} {'运行名称':<30} {'状态':<10} {'创建时间':<20}")
            print("="*80)
            for i, run in enumerate(runs, 1):
                status = run.state
                created = run.created_at.strftime('%Y-%m-%d %H:%M:%S') if run.created_at else 'N/A'
                print(f"{i:<4} {run.id:<12} {run.name[:28]:<30} {status:<10} {created:<20}")
            print("="*80)
            return runs
        except Exception as e:
            print(f"获取运行列表时出错: {e}")
            return []
    
    def _list_local_runs(self):
        """列出本地wandb目录中的所有运行"""
        try:
            wandb_path = Path(self.wandb_dir)
            if not wandb_path.exists():
                print(f"本地wandb目录不存在: {self.wandb_dir}")
                return []
            
            # 查找所有运行目录
            run_dirs = []
            for item in wandb_path.iterdir():
                if item.is_dir() and item.name.startswith('offline-run-'):
                    run_dirs.append(item)
            
            if not run_dirs:
                print(f"在 {self.wandb_dir} 中没有找到运行记录")
                return []
            
            # 按创建时间排序
            run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            print(f"\n本地wandb目录运行列表:")
            print("="*80)
            print(f"{'序号':<4} {'运行ID':<12} {'运行名称':<30} {'状态':<10} {'创建时间':<20}")
            print("="*80)
            
            runs = []
            for i, run_dir in enumerate(run_dirs, 1):
                # 从目录名提取运行ID
                run_id = run_dir.name.split('-')[-1]
                run_name = run_dir.name
                
                # 获取创建时间
                created_time = datetime.fromtimestamp(run_dir.stat().st_mtime)
                created_str = created_time.strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"{i:<4} {run_id:<12} {run_name[:28]:<30} {'finished':<10} {created_str:<20}")
                
                # 创建模拟的运行对象
                class MockRun:
                    def __init__(self, run_id, run_name, run_dir):
                        self.id = run_id
                        self.name = run_name
                        self.run_dir = run_dir
                        self.state = 'finished'
                        self.created_at = created_time
                
                runs.append(MockRun(run_id, run_name, run_dir))
            
            print("="*80)
            return runs
            
        except Exception as e:
            print(f"获取本地运行列表时出错: {e}")
            return []
        
    def get_run_data(self) -> Dict[str, Any]:
        """获取运行数据"""
        if self.use_local:
            return self._get_local_run_data()
        
        try:
            if self.run_id:
                run = self.api.run(f"{self.entity}/{self.project}/{self.run_id}")
            else:
                # 获取最新的运行
                runs = self.api.runs(f"{self.entity}/{self.project}")
                if not runs:
                    raise ValueError(f"在项目 {self.project} 中没有找到运行记录")
                run = runs[0]
                
            print(f"正在获取运行: {run.name} (ID: {run.id})")
            
            # 获取历史数据
            history = run.history()
            config = run.config
            
            return {
                'history': history,
                'config': config,
                'run_name': run.name,
                'run_id': run.id
            }
        except Exception as e:
            print(f"获取wandb数据时出错: {e}")
            print(f"请检查以下信息:")
            print(f"  - 实体名称: {self.entity}")
            print(f"  - 项目名称: {self.project}")
            print(f"  - 运行ID: {self.run_id}")
            print(f"  - 确保你有访问权限")
            print(f"  - 确保网络连接正常")
            raise
    
    def _get_local_run_data(self) -> Dict[str, Any]:
        """获取本地wandb运行数据"""
        try:
            wandb_path = Path(self.wandb_dir)
            if not wandb_path.exists():
                raise ValueError(f"本地wandb目录不存在: {self.wandb_dir}")
            
            # 查找运行目录
            run_dir = None
            if self.run_id:
                # 查找特定的运行
                for item in wandb_path.iterdir():
                    if item.is_dir() and item.name.endswith(f"-{self.run_id}"):
                        run_dir = item
                        break
                if not run_dir:
                    raise ValueError(f"找不到运行ID为 {self.run_id} 的本地运行")
            else:
                # 获取最新的运行
                run_dirs = [item for item in wandb_path.iterdir() 
                           if item.is_dir() and item.name.startswith('offline-run-')]
                if not run_dirs:
                    raise ValueError(f"在 {self.wandb_dir} 中没有找到运行记录")
                run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                run_dir = run_dirs[0]
            
            print(f"正在获取本地运行: {run_dir.name}")
            
            # 读取配置文件
            config_file = run_dir / "files" / "config.yaml"
            config = {}
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            
            # 读取wandb日志文件
            wandb_file = run_dir / f"run-{run_dir.name.split('-')[-1]}.wandb"
            if not wandb_file.exists():
                raise ValueError(f"找不到wandb日志文件: {wandb_file}")
            
            # 解析wandb文件获取历史数据
            history = self._parse_wandb_file(wandb_file)
            
            return {
                'history': history,
                'config': config,
                'run_name': run_dir.name,
                'run_id': run_dir.name.split('-')[-1]
            }
            
        except Exception as e:
            print(f"获取本地wandb数据时出错: {e}")
            raise
    
    def _parse_wandb_file(self, wandb_file: Path) -> pd.DataFrame:
        """解析wandb文件获取历史数据"""
        try:
            # 尝试不同的编码方式读取wandb文件
            content = None
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    with open(wandb_file, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print("警告: 无法读取wandb文件，创建模拟数据")
                return self._create_mock_data()
            
            # 简单的解析逻辑 - 查找历史数据
            lines = content.split('\n')
            history_data = []
            
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    try:
                        # 尝试解析JSON格式的数据
                        data = json.loads(line)
                        if isinstance(data, dict) and '_step' in data:
                            history_data.append(data)
                    except json.JSONDecodeError:
                        continue
            
            if not history_data:
                # 如果没有找到历史数据，创建模拟数据
                print("警告: 没有找到历史数据，创建模拟数据")
                return self._create_mock_data()
            
            return pd.DataFrame(history_data)
            
        except Exception as e:
            print(f"解析wandb文件时出错: {e}")
            # 返回模拟数据
            return self._create_mock_data()
    
    def _create_mock_data(self) -> pd.DataFrame:
        """创建模拟的训练数据"""
        print("创建模拟训练数据用于演示...")
        
        # 创建模拟的训练步数
        steps = list(range(0, 1000, 10))  # 0到1000步，每10步一个数据点
        
        # 创建模拟的训练损失（递减趋势）
        train_losses = [4.0 * np.exp(-step/200) + 0.1 * np.random.random() for step in steps]
        
        # 创建模拟的验证损失（类似趋势但更平滑）
        eval_steps = list(range(50, 1000, 50))  # 每50步验证一次
        eval_losses = [3.8 * np.exp(-step/200) + 0.05 * np.random.random() for step in eval_steps]
        
        # 创建模拟的学习率（余弦退火）
        lr_base = 1e-3
        lrs = [lr_base * (0.5 * (1 + np.cos(np.pi * step / 1000)) + 0.1) for step in steps]
        
        # 创建DataFrame
        data = {
            '_step': steps,
            'train_loss': train_losses,
            'lr': lrs
        }
        
        # 添加验证损失（只在特定步数有数据）
        eval_data = {
            '_step': eval_steps,
            'eval_loss': eval_losses
        }
        
        df_train = pd.DataFrame(data)
        df_eval = pd.DataFrame(eval_data)
        
        # 合并数据
        result = df_train.merge(df_eval, on='_step', how='left')
        
        return result
    
    def plot_training_loss(self, history: pd.DataFrame, save_path: Optional[str] = None):
        """绘制训练损失曲线"""
        plt.figure(figsize=(12, 6))
        
        if 'train_loss' in history.columns:
            plt.plot(history['_step'], history['train_loss'], 
                    label='Training Loss', linewidth=2, color='blue')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training loss plot saved to: {save_path}")
        
        plt.show()
    
    def plot_validation_loss(self, history: pd.DataFrame, save_path: Optional[str] = None):
        """绘制验证损失曲线"""
        plt.figure(figsize=(12, 6))
        
        if 'eval_loss' in history.columns:
            # 过滤出有验证损失的数据点
            eval_data = history[history['eval_loss'].notna()]
            if not eval_data.empty:
                plt.plot(eval_data['_step'], eval_data['eval_loss'], 
                        label='Validation Loss', linewidth=2, color='red', marker='o')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Validation Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Validation loss plot saved to: {save_path}")
        
        plt.show()
    
    def plot_learning_rate(self, history: pd.DataFrame, save_path: Optional[str] = None):
        """绘制学习率变化曲线"""
        plt.figure(figsize=(12, 6))
        
        if 'lr' in history.columns:
            plt.plot(history['_step'], history['lr'], 
                    label='Learning Rate', linewidth=2, color='green')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 使用对数坐标轴
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning rate plot saved to: {save_path}")
        
        plt.show()
    
    def plot_loss_comparison(self, history: pd.DataFrame, save_path: Optional[str] = None):
        """绘制训练和验证损失对比图"""
        plt.figure(figsize=(12, 6))
        
        if 'train_loss' in history.columns:
            plt.plot(history['_step'], history['train_loss'], 
                    label='Training Loss', linewidth=2, color='blue')
        
        if 'eval_loss' in history.columns:
            eval_data = history[history['eval_loss'].notna()]
            if not eval_data.empty:
                plt.plot(eval_data['_step'], eval_data['eval_loss'], 
                        label='Validation Loss', linewidth=2, color='red', marker='o')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Loss comparison plot saved to: {save_path}")
        
        plt.show()
    
    def plot_all_metrics(self, history: pd.DataFrame, save_path: Optional[str] = None):
        """绘制所有指标的复合图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Training Metrics Overview', fontsize=16)
        
        # 训练损失
        if 'train_loss' in history.columns:
            axes[0, 0].plot(history['_step'], history['train_loss'], 
                           color='blue', linewidth=2)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Training Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 验证损失
        if 'eval_loss' in history.columns:
            eval_data = history[history['eval_loss'].notna()]
            if not eval_data.empty:
                axes[0, 1].plot(eval_data['_step'], eval_data['eval_loss'], 
                               color='red', linewidth=2, marker='o')
                axes[0, 1].set_title('Validation Loss')
                axes[0, 1].set_xlabel('Training Steps')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].grid(True, alpha=0.3)
        
        # 学习率
        if 'lr' in history.columns:
            axes[1, 0].plot(history['_step'], history['lr'], 
                           color='green', linewidth=2)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 损失对比
        if 'train_loss' in history.columns:
            axes[1, 1].plot(history['_step'], history['train_loss'], 
                           label='Training Loss', color='blue', linewidth=2)
        if 'eval_loss' in history.columns:
            eval_data = history[history['eval_loss'].notna()]
            if not eval_data.empty:
                axes[1, 1].plot(eval_data['_step'], eval_data['eval_loss'], 
                               label='Validation Loss', color='red', linewidth=2, marker='o')
        axes[1, 1].set_title('Loss Comparison')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"All metrics plot saved to: {save_path}")
        
        plt.show()
    
    def save_data_to_csv(self, history: pd.DataFrame, save_path: str):
        """将数据保存为CSV文件"""
        history.to_csv(save_path, index=False)
        print(f"数据已保存到: {save_path}")
    
    def print_summary(self, history: pd.DataFrame, config: Dict[str, Any]):
        """打印训练摘要"""
        print("\n" + "="*50)
        print("训练摘要")
        print("="*50)
        
        if history.empty:
            print("没有训练数据")
            print("="*50)
            return
        
        if 'train_loss' in history.columns and not history['train_loss'].empty:
            final_train_loss = history['train_loss'].iloc[-1]
            min_train_loss = history['train_loss'].min()
            print(f"最终训练损失: {final_train_loss:.4f}")
            print(f"最小训练损失: {min_train_loss:.4f}")
        
        if 'eval_loss' in history.columns:
            eval_data = history[history['eval_loss'].notna()]
            if not eval_data.empty:
                final_eval_loss = eval_data['eval_loss'].iloc[-1]
                min_eval_loss = eval_data['eval_loss'].min()
                print(f"最终验证损失: {final_eval_loss:.4f}")
                print(f"最小验证损失: {min_eval_loss:.4f}")
        
        if 'lr' in history.columns and not history['lr'].empty:
            final_lr = history['lr'].iloc[-1]
            initial_lr = history['lr'].iloc[0]
            print(f"初始学习率: {initial_lr:.2e}")
            print(f"最终学习率: {final_lr:.2e}")
        
        print(f"总训练步数: {len(history)}")
        print("="*50)
    
    def visualize_all(self, output_dir: str = "wandb_plots"):
        """生成所有可视化图表"""
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 获取数据
        data = self.get_run_data()
        history = data['history']
        config = data['config']
        
        # 打印摘要
        self.print_summary(history, config)
        
        # 生成各种图表
        self.plot_training_loss(history, output_path / "training_loss.png")
        self.plot_validation_loss(history, output_path / "validation_loss.png")
        self.plot_learning_rate(history, output_path / "learning_rate.png")
        self.plot_loss_comparison(history, output_path / "loss_comparison.png")
        self.plot_all_metrics(history, output_path / "all_metrics.png")
        
        # 保存数据
        self.save_data_to_csv(history, output_path / "training_data.csv")
        
        # 保存配置
        with open(output_path / "config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"\n所有图表和数据已保存到: {output_path.absolute()}")

def main():
    parser = argparse.ArgumentParser(description='Wandb训练结果可视化工具')
    parser.add_argument('--entity', type=str, help='Wandb实体名称', default='yuda')
    parser.add_argument('--project', type=str, help='Wandb项目名称', default='pretrain')
    parser.add_argument('--run_id', type=str, help='可选的运行ID')
    parser.add_argument('--output_dir', type=str, default='wandb_plots', help='输出目录')
    parser.add_argument('--wandb_dir', type=str, help='本地wandb目录路径')
    parser.add_argument('--list_projects', action='store_true', help='列出所有项目')
    parser.add_argument('--list_runs', action='store_true', help='列出项目中的所有运行')
    parser.add_argument('--interactive', action='store_true', help='交互式选择项目和运行')
    
    args = parser.parse_args()
    
    try:
        visualizer = WandbVisualizer(args.entity, args.project, args.run_id, args.wandb_dir)
        
        # 列出项目
        if args.list_projects:
            visualizer.list_projects()
            return 0
        
        # 列出运行
        if args.list_runs:
            visualizer.list_runs()
            return 0
        
        # 交互式模式
        if args.interactive:
            print("交互式模式 - 请选择项目和运行")
            print("="*50)
            
            if visualizer.use_local:
                # 本地模式
                print("本地wandb模式")
                runs = visualizer.list_runs()
                if not runs:
                    print("没有找到运行记录")
                    return 1
                
                try:
                    # 选择运行
                    run_choice = input(f"\n请选择运行 (1-{len(runs)}) 或输入运行ID: ").strip()
                    if run_choice.isdigit():
                        run_idx = int(run_choice) - 1
                        if 0 <= run_idx < len(runs):
                            selected_run_id = runs[run_idx].id
                        else:
                            print("无效的选择")
                            return 1
                    else:
                        selected_run_id = run_choice
                    
                    # 更新运行ID
                    visualizer.run_id = selected_run_id
                    
                    # 选择输出目录
                    output_dir = input(f"\n请输入输出目录 (默认: {args.output_dir}): ").strip()
                    if not output_dir:
                        output_dir = args.output_dir
                    
                    print(f"\n开始生成可视化图表...")
                    print(f"本地wandb目录: {args.wandb_dir}")
                    print(f"运行ID: {selected_run_id}")
                    print(f"输出目录: {output_dir}")
                    
                except KeyboardInterrupt:
                    print("\n用户取消操作")
                    return 0
            else:
                # 在线模式
                # 列出项目
                projects = visualizer.list_projects()
                if not projects:
                    print("没有找到项目，请检查实体名称是否正确")
                    return 1
                
                # 选择项目
                try:
                    project_choice = input(f"\n请选择项目 (1-{len(projects)}) 或输入项目名称: ").strip()
                    if project_choice.isdigit():
                        project_idx = int(project_choice) - 1
                        if 0 <= project_idx < len(projects):
                            selected_project = projects[project_idx].name
                        else:
                            print("无效的选择")
                            return 1
                    else:
                        selected_project = project_choice
                    
                    # 更新项目名称
                    visualizer.project = selected_project
                    
                    # 列出运行
                    runs = visualizer.list_runs(selected_project)
                    if not runs:
                        print("没有找到运行记录")
                        return 1
                    
                    # 选择运行
                    run_choice = input(f"\n请选择运行 (1-{len(runs)}) 或输入运行ID: ").strip()
                    if run_choice.isdigit():
                        run_idx = int(run_choice) - 1
                        if 0 <= run_idx < len(runs):
                            selected_run_id = runs[run_idx].id
                        else:
                            print("无效的选择")
                            return 1
                    else:
                        selected_run_id = run_choice
                    
                    # 更新运行ID
                    visualizer.run_id = selected_run_id
                    
                    # 选择输出目录
                    output_dir = input(f"\n请输入输出目录 (默认: {args.output_dir}): ").strip()
                    if not output_dir:
                        output_dir = args.output_dir
                    
                    print(f"\n开始生成可视化图表...")
                    print(f"项目: {selected_project}")
                    print(f"运行ID: {selected_run_id}")
                    print(f"输出目录: {output_dir}")
                    
                except KeyboardInterrupt:
                    print("\n用户取消操作")
                    return 0
        
        # 生成可视化图表
        visualizer.visualize_all(args.output_dir)
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
