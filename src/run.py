#!/usr/bin/env python3
"""
自动运行脚本 - 导入并运行所有模型
用于DBPs（消毒副产品）预测的综合模型评估
"""

import os
import sys
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 导入自定义模型类
from models import ComprehensiveModelEvaluator

# 设置matplotlib后端以兼容服务器环境
plt.switch_backend('Agg')

def main():
    """主函数 - 运行所有模型并生成完整报告"""

    # 切换到项目根目录（src/ 的上一级）
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    print("="*80)
    print("    DBPs预测模型综合评估系统 (增强版)")
    print("    Enhanced Comprehensive DBPs Prediction Model Evaluation")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = time.time()

    try:
        # 检查数据文件是否存在
        data_path = os.path.join('data', 'data.csv')
        if not os.path.exists(data_path):
            print("❌ 错误: 未找到数据文件 'data/data.csv'")
            print("请确保数据文件存在于 data/ 目录中。")
            return False

        # 创建综合模型评估器
        print("📊 初始化增强版综合模型评估器...")
        evaluator = ComprehensiveModelEvaluator(
            data_path=data_path,
            results_dir='results'
        )
        
        print("✅ 评估器初始化完成")
        print()
        
        # 运行所有模型 (深度学习模型使用300个epochs)
        print("🚀 开始训练和评估所有模型 (深度学习模型: 300 epochs)...")
        print("-" * 60)
        
        all_results = evaluator.run_all_models(include_explainable=True, run_ablation=True)
        
        if not all_results:
            print("❌ 警告: 没有成功训练的模型")
            return False
        
        print(f"✅ 成功训练了 {len(all_results)} 个模型")
        print()
        
        # 打印简要结果总结
        print("="*80)
        print("模型性能排名 (按 Test R² 排序)")
        print("="*80)
        
        # 按R²得分排序并显示
        sorted_results = sorted(all_results.items(), 
                               key=lambda x: x[1]['Test R²'], 
                               reverse=True)
        
        print(f"{'排名':<4} {'模型名称':<25} {'Test R²':<10} {'Test MSE':<12} {'Test MAE':<12}")
        print("-" * 75)
        
        for rank, (model_name, metrics) in enumerate(sorted_results, 1):
            print(f"{rank:<4} {model_name:<25} {metrics['Test R²']:<10.4f} "
                  f"{metrics['Test MSE']:<12.4f} {metrics['Test MAE']:<12.4f}")
        
        print()
        print("="*80)
        print("最佳模型总结")
        print("="*80)
        
        # 找到各指标最佳模型
        best_r2_model = max(all_results.keys(), key=lambda x: all_results[x]['Test R²'])
        best_mse_model = min(all_results.keys(), key=lambda x: all_results[x]['Test MSE'])
        best_mae_model = min(all_results.keys(), key=lambda x: all_results[x]['Test MAE'])
        
        print(f"🏆 最佳R²模型: {best_r2_model}")
        print(f"   - Test R² = {all_results[best_r2_model]['Test R²']:.4f}")
        print(f"   - Test MSE = {all_results[best_r2_model]['Test MSE']:.4f}")
        print(f"   - Test MAE = {all_results[best_r2_model]['Test MAE']:.4f}")
        print()
        
        print(f"🎯 最佳MSE模型: {best_mse_model}")
        print(f"   - Test R² = {all_results[best_mse_model]['Test R²']:.4f}")
        print(f"   - Test MSE = {all_results[best_mse_model]['Test MSE']:.4f}")
        print(f"   - Test MAE = {all_results[best_mse_model]['Test MAE']:.4f}")
        print()
        
        print(f"🎪 最佳MAE模型: {best_mae_model}")
        print(f"   - Test R² = {all_results[best_mae_model]['Test R²']:.4f}")
        print(f"   - Test MSE = {all_results[best_mae_model]['Test MSE']:.4f}")
        print(f"   - Test MAE = {all_results[best_mae_model]['Test MAE']:.4f}")
        print()
        
        # 打印5折交叉验证结果
        if hasattr(evaluator, 'cv_results') and evaluator.cv_results:
            print()
            print("="*80)
            print("5折交叉验证结果 (CV R² Mean ± Std)")
            print("="*80)
            cv_sorted = sorted(evaluator.cv_results.items(),
                               key=lambda x: x[1]['CV R² Mean'], reverse=True)
            print(f"{'排名':<4} {'模型名称':<30} {'CV R² Mean':<14} {'CV R² Std':<12} {'CV MSE Mean':<14}")
            print("-" * 80)
            for rank, (mname, mvals) in enumerate(cv_sorted, 1):
                print(f"{rank:<4} {mname:<30} {mvals['CV R² Mean']:<14.4f} "
                      f"{mvals['CV R² Std']:<12.4f} {mvals['CV MSE Mean']:<14.4f}")

        # 打印统计检验结果
        if hasattr(evaluator, 'stat_results') and evaluator.stat_results:
            print()
            print("="*80)
            print("Wilcoxon配对显著性检验结果 (vs 最佳CV模型)")
            print("="*80)
            stat_sorted = sorted(evaluator.stat_results.items(),
                                 key=lambda x: x[1]['p_value'])
            print(f"{'模型':<30} {'p值':<10} {'显著性':<10} {'ΔR²':<10}")
            print("-" * 65)
            for mname, res in stat_sorted:
                sig = '* (p<0.05)' if res['significant'] else 'ns'
                print(f"{mname:<30} {res['p_value']:<10.4f} {sig:<10} {res['delta_R2']:<10.4f}")

        # 计算运行时间
        end_time = time.time()
        total_time = end_time - start_time

        print()
        print("="*80)
        print("输出文件")
        print("="*80)
        print("📁 results/ - 所有结果文件目录")
        print("   📈 model_results_*.csv - 单次分割数值结果")
        print("   📊 model_comparison_*.png - 模型性能比较图")
        print("   📋 cross_validation_results_*.csv - 5折CV结果")
        print("   📊 cv_comparison_*.png - CV结果对比图")
        print("   📋 statistical_tests_*.csv - Wilcoxon显著性检验")
        print("   📝 model_evaluation_report_*.md - 详细评估报告")
        print("   📁 advanced_analysis/ - 特征重要性、残差分析、相关性分析")
        print("   📁 ablation_results/ - 特征消融研究结果")
        print()

        print("="*80)
        print("执行完成")
        print("="*80)
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        print()
        print("评估完成! 请查看results目录中的详细结果。")
        print()
        print("本次新增功能:")
        print("✅ 修复数据泄漏问题（测试集不再混入训练集）")
        print("✅ 加入TabNet模型对比")
        print("✅ 5折交叉验证 (CV R² Mean ± Std)")
        print("✅ Wilcoxon配对显著性检验")
        print("✅ 深度学习模型训练300个epochs")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断了执行")
        return False
        
    except Exception as e:
        print(f"\n❌ 执行过程中发生错误: {str(e)}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        return False


def print_system_info():
    """打印系统信息"""
    print("系统信息:")
    print(f"Python版本: {sys.version}")
    print(f"工作目录: {os.getcwd()}")
    
    # 检查重要的库
    libraries = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'sklearn': 'scikit-learn',
        'torch': 'PyTorch',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    print("\n已安装的库:")
    for lib_name, display_name in libraries.items():
        try:
            if lib_name == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                lib = __import__(lib_name)
                version = getattr(lib, '__version__', 'Unknown')
            print(f"✅ {display_name}: {version}")
        except ImportError:
            print(f"❌ {display_name}: 未安装")
    
    # 检查KAN库
    try:
        import kan
        print(f"✅ KAN: Available")
    except ImportError:
        print(f"⚠️  KAN: 不可用 (将跳过KAN模型)")
    
    print()


if __name__ == "__main__":
    # 打印系统信息
    print_system_info()
    
    # 运行主程序
    success = main()
    
    # 根据执行结果设置退出码
    sys.exit(0 if success else 1) 