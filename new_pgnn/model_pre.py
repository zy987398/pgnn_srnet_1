import os
import argparse
import warnings
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 导入自定义模块
from models import MaterialCrackPGNN
from utils import prepare_material_data

# 忽略警告
warnings.filterwarnings("ignore")


class CrackDepthPredictor:
    """裂纹深度预测器"""
    
    def __init__(self, model_path, train_data_path, device=None):
        """
        初始化预测器
        
        参数:
            model_path: 训练好的模型文件路径 (.pth)
            train_data_path: 训练数据路径 (用于获取缩放器和标签编码器)
            device: 计算设备 ('cuda' 或 'cpu')
        """
        # 修正模型路径 - 添加自动路径检测
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            # 尝试相对于当前目录的父目录路径
            parent_model_path = os.path.join("..", "results", os.path.basename(model_path))
            if os.path.exists(parent_model_path):
                model_path = parent_model_path
        
        self.model_path = model_path
        self.train_data_path = train_data_path
        
        # 设置设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 验证文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.exists(train_data_path):
            raise FileNotFoundError(f"训练数据文件不存在: {train_data_path}")
        
        # 加载训练数据以获取缩放器和标签编码器
        print(f"加载训练数据: {train_data_path}")
        try:
            self.train_df = pd.read_csv(train_data_path)
            self.train_data = prepare_material_data(self.train_df)
            
            self.scaler = self.train_data["scaler"]
            self.label_encoder = self.train_data["label_encoder"]
            self.num_materials = self.train_data["num_materials"]
            
        except Exception as e:
            raise ValueError(f"加载训练数据失败: {str(e)}")
        
        print(f"材料种类数: {self.num_materials}")
        print(f"支持的材料: {list(self.label_encoder.classes_)}")
        
        # 加载模型
        self._load_model()
    
    def _handle_unknown_materials_by_properties(self, input_df):
        """
        基于物理性质匹配未知材料的防御机制
        """
        # 检查材料种类
        unknown_materials = set(input_df["材料种类"]) - set(self.label_encoder.classes_)
        
        if unknown_materials:
            print(f"\n警告: 以下材料未在训练集中出现，将基于物理性质匹配:")
            for mat in unknown_materials:
                print(f"  - {mat}")
            
            # 计算训练集中每种材料的平均物理性质
            train_material_properties = self.train_df.groupby("材料种类").agg({
                "莫氏硬度": ["mean", "std", "count"],
                "弹性模量": ["mean", "std", "count"],
                "断裂韧性": ["mean", "std", "count"],
                "晶粒粒径": ["mean", "std", "count"]
            }).reset_index()
            
            # 展平多级列索引
            train_material_properties.columns = [
                '材料种类',
                '莫氏硬度_mean', '莫氏硬度_std', '莫氏硬度_count',
                '弹性模量_mean', '弹性模量_std', '弹性模量_count',
                '断裂韧性_mean', '断裂韧性_std', '断裂韧性_count',
                '晶粒粒径_mean', '晶粒粒径_std', '晶粒粒径_count'
            ]
            
            # 处理标准差为0的情况（添加小的值避免除零）
            for col in ['莫氏硬度_std', '弹性模量_std', '断裂韧性_std', '晶粒粒径_std']:
                train_material_properties[col] = train_material_properties[col].fillna(1e-6)
                train_material_properties[col] = np.maximum(train_material_properties[col], 1e-6)
            
            # 为每个未知材料找到最相似的已知材料
            replacement_map = {}
            
            for unknown_mat in unknown_materials:
                print(f"\n  处理未知材料: {unknown_mat}")
                
                # 获取该未知材料的所有样本
                unknown_samples = input_df[input_df["材料种类"] == unknown_mat]
                
                # 计算未知材料的平均物理性质
                unknown_properties = {
                    "莫氏硬度": unknown_samples["莫氏硬度"].mean(),
                    "弹性模量": unknown_samples["弹性模量"].mean(), 
                    "断裂韧性": unknown_samples["断裂韧性"].mean(),
                    "晶粒粒径": unknown_samples["晶粒粒径"].mean()
                }
                
                print(f"    未知材料平均性质:")
                for prop, value in unknown_properties.items():
                    if not pd.isna(value):
                        print(f"      {prop}: {value:.4f}")
                    else:
                        print(f"      {prop}: 缺失")
                
                # 计算与每种已知材料的相似度
                similarities = []
                
                for _, known_material in train_material_properties.iterrows():
                    material_name = known_material["材料种类"]
                    
                    # 计算标准化的欧氏距离
                    weighted_distance = 0
                    valid_properties = 0
                    property_weights = {
                        "莫氏硬度": 1.0,     # 硬度权重
                        "弹性模量": 0.8,     # 弹性模量权重（数值较大，降低权重）
                        "断裂韧性": 1.2,     # 韧性权重（重要性较高）
                        "晶粒粒径": 0.6      # 晶粒大小权重
                    }
                    
                    for prop in ["莫氏硬度", "弹性模量", "断裂韧性", "晶粒粒径"]:
                        unknown_val = unknown_properties[prop]
                        
                        if not pd.isna(unknown_val):
                            known_mean = known_material[f"{prop}_mean"]
                            known_std = known_material[f"{prop}_std"]
                            
                            # 标准化距离 = |未知值 - 已知均值| / 已知标准差
                            if known_std > 0:
                                normalized_dist = abs(unknown_val - known_mean) / known_std
                            else:
                                # 如果标准差为0，使用绝对差值的相对比例
                                if known_mean != 0:
                                    normalized_dist = abs(unknown_val - known_mean) / abs(known_mean)
                                else:
                                    normalized_dist = abs(unknown_val)
                            
                            # 应用权重
                            weighted_distance += property_weights[prop] * normalized_dist ** 2
                            valid_properties += property_weights[prop]
                    
                    if valid_properties > 0:
                        # 计算加权平均距离
                        final_distance = np.sqrt(weighted_distance / valid_properties)
                        
                        # 转换为相似度 (距离越小相似度越高)
                        similarity = 1 / (1 + final_distance)
                        
                        similarities.append({
                            "材料": material_name,
                            "相似度": similarity,
                            "距离": final_distance,
                            "样本数": known_material["莫氏硬度_count"]
                        })
                
                # 按相似度排序
                similarities.sort(key=lambda x: x["相似度"], reverse=True)
                
                # 选择最相似的材料，但要考虑样本数的可靠性
                best_candidates = similarities[:3]  # 前3个候选
                
                # 在前3个中选择样本数最多的（更可靠）
                if len(best_candidates) > 1:
                    # 如果最佳候选的相似度明显高于其他候选（>0.1差距），直接选择
                    if best_candidates[0]["相似度"] - best_candidates[1]["相似度"] > 0.1:
                        best_match = best_candidates[0]
                    else:
                        # 否则在前几个中选择样本数最多的
                        best_match = max(best_candidates, key=lambda x: x["样本数"])
                else:
                    best_match = best_candidates[0]
                
                replacement_map[unknown_mat] = best_match["材料"]
                
                print(f"    匹配结果:")
                print(f"      最佳匹配: {best_match['材料']}")
                print(f"      相似度: {best_match['相似度']:.4f}")
                print(f"      标准化距离: {best_match['距离']:.4f}")
                print(f"      训练样本数: {best_match['样本数']}")
                
                print(f"    备选方案 (前3名):")
                for i, candidate in enumerate(best_candidates):
                    print(f"      {i+1}. {candidate['材料']} (相似度: {candidate['相似度']:.4f}, 样本数: {candidate['样本数']})")
            
            # 执行替换
            input_df["材料种类"] = input_df["材料种类"].map(
                lambda x: replacement_map.get(x, x)
            )
            
            print(f"\n材料替换完成:")
            for unknown, known in replacement_map.items():
                print(f"  {unknown} -> {known}")
        
        return input_df
    
    def _find_similar_material_by_properties(self, target_properties):
        """
        根据单个样本的物理性质找到最相似的已知材料
        
        参数:
            target_properties: dict, 包含物理性质的字典
        
        返回:
            最相似的材料名称和相似度
        """
        # 计算训练集中每种材料的平均物理性质
        train_material_properties = self.train_df.groupby("材料种类").agg({
            "莫氏硬度": ["mean", "std"],
            "弹性模量": ["mean", "std"],
            "断裂韧性": ["mean", "std"],
            "晶粒粒径": ["mean", "std"]
        }).reset_index()
        
        # 展平多级列索引
        train_material_properties.columns = [
            '材料种类',
            '莫氏硬度_mean', '莫氏硬度_std',
            '弹性模量_mean', '弹性模量_std',
            '断裂韧性_mean', '断裂韧性_std',
            '晶粒粒径_mean', '晶粒粒径_std'
        ]
        
        # 处理标准差为0的情况
        for col in ['莫氏硬度_std', '弹性模量_std', '断裂韧性_std', '晶粒粒径_std']:
            train_material_properties[col] = train_material_properties[col].fillna(1e-6)
            train_material_properties[col] = np.maximum(train_material_properties[col], 1e-6)
        
        best_similarity = 0
        best_material = list(self.label_encoder.classes_)[0]  # 默认材料
        
        property_weights = {
            "莫氏硬度": 1.0,
            "弹性模量": 0.8,
            "断裂韧性": 1.2,
            "晶粒粒径": 0.6
        }
        
        for _, known_material in train_material_properties.iterrows():
            material_name = known_material["材料种类"]
            
            weighted_distance = 0
            valid_properties = 0
            
            for prop in ["莫氏硬度", "弹性模量", "断裂韧性", "晶粒粒径"]:
                if prop in target_properties and not pd.isna(target_properties[prop]):
                    target_val = target_properties[prop]
                    known_mean = known_material[f"{prop}_mean"]
                    known_std = known_material[f"{prop}_std"]
                    
                    # 标准化距离
                    if known_std > 0:
                        normalized_dist = abs(target_val - known_mean) / known_std
                    else:
                        if known_mean != 0:
                            normalized_dist = abs(target_val - known_mean) / abs(known_mean)
                        else:
                            normalized_dist = abs(target_val)
                    
                    weighted_distance += property_weights[prop] * normalized_dist ** 2
                    valid_properties += property_weights[prop]
            
            if valid_properties > 0:
                final_distance = np.sqrt(weighted_distance / valid_properties)
                similarity = 1 / (1 + final_distance)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_material = material_name
        
        return best_material, best_similarity
    
    def _load_model(self):
        """加载训练好的模型"""
        print(f"\n加载模型: {self.model_path}")
        
        try:
            # 检查模型文件内容
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 如果保存的是完整checkpoint (包含模型配置)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # 获取模型配置
                model_config = checkpoint.get('model_config', {})
                print(f"检测到模型配置: {model_config}")
                
                # 使用保存的配置或默认配置
                self.model = MaterialCrackPGNN(
                    num_materials=self.num_materials,
                    material_embed_dim=model_config.get('material_embed_dim', 8),
                    latent_dim=model_config.get('latent_dim', 48),
                    dropout_rate=model_config.get('dropout_rate', 0.35),
                    symbolic_type=model_config.get('symbolic_type', 'advanced')
                )
                
            else:
                # 如果只保存了state_dict，使用默认配置
                print("使用默认模型配置")
                state_dict = checkpoint
                self.model = MaterialCrackPGNN(
                    num_materials=self.num_materials,
                    material_embed_dim=8,
                    latent_dim=48,
                    dropout_rate=0.35,
                    symbolic_type="advanced"
                )
            
            # 加载模型参数
            self.model.load_state_dict(state_dict)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("模型加载成功!")
            
            # 打印模型参数总数
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"模型参数总数: {total_params:,}")
            
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")
    
    def predict(self, input_data_path, output_path=None, use_mc_dropout=False, n_samples=10):
        """
        对新数据进行预测
        
        参数:
            input_data_path: 输入CSV文件路径
            output_path: 输出CSV文件路径 (可选)
            use_mc_dropout: 是否使用MC Dropout估计不确定度
            n_samples: MC Dropout采样次数
            
        返回:
            包含预测结果的DataFrame
        """
        print(f"\n{'='*60}")
        print(f"开始预测")
        print(f"{'='*60}")
        
        # 验证输入文件
        if not os.path.exists(input_data_path):
            raise FileNotFoundError(f"输入文件不存在: {input_data_path}")
        
        # 读取输入数据
        print(f"读取输入数据: {input_data_path}")
        try:
            input_df = pd.read_csv(input_data_path)
        except Exception as e:
            raise ValueError(f"读取输入文件失败: {str(e)}")
            
        print(f"输入数据大小: {len(input_df)} 样本")
        
        # 检查必需的列
        required_cols = ["材料种类", "莫氏硬度", "弹性模量", "断裂韧性", 
                        "晶粒粒径", "压溃压力", "磨粒速度"]
        missing_cols = [col for col in required_cols if col not in input_df.columns]
        
        if missing_cols:
            raise ValueError(f"输入数据缺少必需的列: {missing_cols}")
        
        # 检查并处理NaN值 - 修正逻辑
        if input_df[required_cols].isnull().any().any():
            print("警告: 输入数据包含缺失值，将使用均值填充...")
            for col in required_cols:
                if col != "材料种类" and input_df[col].isnull().any():
                    # 使用训练数据的均值填充
                    if col in self.train_df.columns:
                        fill_value = self.train_df[col].mean()
                    else:
                        fill_value = input_df[col].mean()
                    input_df[col] = input_df[col].fillna(fill_value)
                    print(f"  - {col}: 填充值 {fill_value:.4f}")
        
        # 使用改进的基于物理性质的材料匹配
        input_df = self._handle_unknown_materials_by_properties(input_df)
        
        # 准备数据
        try:
            input_data = prepare_material_data(
                input_df,
                scaler=self.scaler,
                le=self.label_encoder,
                augment=False
            )
        except Exception as e:
            raise ValueError(f"数据准备失败: {str(e)}")
        
        # 转换为张量
        material_ids = torch.LongTensor(input_data["material_ids"]).to(self.device)
        numeric_raw = torch.FloatTensor(input_data["numeric_raw"]).to(self.device)
        
        # 进行预测
        print("\n开始预测...")
        
        try:
            # 检查模型是否有MC Dropout方法
            if use_mc_dropout and hasattr(self.model, 'predict_with_uncertainty'):
                print(f"使用MC Dropout (采样次数: {n_samples})...")
                predictions, total_uncertainty, aleatoric_unc, epistemic_unc = \
                    self.model.predict_with_uncertainty(
                        material_ids, numeric_raw, n_samples=n_samples
                    )
                
                # 转换为numpy
                predictions_np = predictions.cpu().squeeze().numpy()
                total_uncertainty_np = total_uncertainty.cpu().squeeze().numpy()
                aleatoric_unc_np = aleatoric_unc.cpu().squeeze().numpy()
                epistemic_unc_np = epistemic_unc.cpu().squeeze().numpy()
                
            else:
                with torch.no_grad():
                    predictions, uncertainties, _, _ = self.model(material_ids, numeric_raw)
                
                # 转换为numpy
                predictions_np = predictions.cpu().squeeze().numpy()
                total_uncertainty_np = uncertainties.cpu().squeeze().numpy()
                aleatoric_unc_np = total_uncertainty_np
                epistemic_unc_np = np.zeros_like(aleatoric_unc_np)
                
        except Exception as e:
            raise RuntimeError(f"模型预测失败: {str(e)}")
        
        # 确保是一维数组
        if predictions_np.ndim == 0:
            predictions_np = np.array([predictions_np])
        if total_uncertainty_np.ndim == 0:
            total_uncertainty_np = np.array([total_uncertainty_np])
            aleatoric_unc_np = np.array([aleatoric_unc_np])
            epistemic_unc_np = np.array([epistemic_unc_np])
        
        # 创建结果DataFrame
        result_df = input_df.copy()
        result_df["预测裂纹深度"] = predictions_np
        result_df["不确定度"] = np.sqrt(np.maximum(total_uncertainty_np, 1e-8))  # 确保非负
        
        if use_mc_dropout:
            result_df["随机不确定度"] = np.sqrt(np.maximum(aleatoric_unc_np, 1e-8))
            result_df["认知不确定度"] = np.sqrt(np.maximum(epistemic_unc_np, 1e-8))
        
        # 打印预测统计
        print(f"\n预测完成!")
        print(f"{'='*60}")
        print(f"预测统计:")
        print(f"  - 样本数: {len(predictions_np)}")
        print(f"  - 平均裂纹深度: {predictions_np.mean():.4f}")
        print(f"  - 裂纹深度范围: [{predictions_np.min():.4f}, {predictions_np.max():.4f}]")
        print(f"  - 平均不确定度: {np.sqrt(np.maximum(total_uncertainty_np, 1e-8)).mean():.4f}")
        print(f"{'='*60}")
        
        # 如果输入数据包含真实值,计算指标
        if "亚表面裂纹深度" in input_df.columns:
            try:
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                
                true_values = input_df["亚表面裂纹深度"].values
                r2 = r2_score(true_values, predictions_np)
                rmse = np.sqrt(mean_squared_error(true_values, predictions_np))
                mae = mean_absolute_error(true_values, predictions_np)
                
                print(f"\n评估指标 (与真实值对比):")
                print(f"  - R² Score: {r2:.4f}")
                print(f"  - RMSE: {rmse:.4f}")
                print(f"  - MAE: {mae:.4f}")
                print(f"{'='*60}")
            except Exception as e:
                print(f"计算评估指标时出错: {str(e)}")
        
        # 保存结果
        if output_path:
            try:
                # 创建输出目录
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"\n预测结果已保存至: {output_path}")
            except Exception as e:
                print(f"保存结果时出错: {str(e)}")
        
        return result_df
    
    def predict_single(self, material_type, hardness, elastic_modulus, 
                      fracture_toughness, grain_size, pressure, velocity):
        """
        对单个样本进行预测（增加基于物理性质的防御机制）
        
        参数:
            material_type: 材料种类
            hardness: 莫氏硬度
            elastic_modulus: 弹性模量
            fracture_toughness: 断裂韧性
            grain_size: 晶粒粒径
            pressure: 压溃压力
            velocity: 磨粒速度
            
        返回:
            (预测裂纹深度, 不确定度)
        """
        # 检查材料种类是否已知
        if material_type not in self.label_encoder.classes_:
            print(f"警告: 材料 '{material_type}' 未在训练集中出现")
            
            # 基于物理性质找最相似的材料
            target_properties = {
                "莫氏硬度": hardness,
                "弹性模量": elastic_modulus,
                "断裂韧性": fracture_toughness,
                "晶粒粒径": grain_size
            }
            
            replacement, similarity = self._find_similar_material_by_properties(target_properties)
            
            print(f"基于物理性质匹配:")
            print(f"  目标材料性质:")
            for prop, value in target_properties.items():
                if not pd.isna(value):
                    print(f"    {prop}: {value}")
            print(f"  最佳匹配材料: {replacement}")
            print(f"  相似度: {similarity:.4f}")
            
            material_type = replacement
        
        # 创建临时DataFrame
        temp_df = pd.DataFrame({
            "材料种类": [material_type],
            "莫氏硬度": [hardness],
            "弹性模量": [elastic_modulus],
            "断裂韧性": [fracture_toughness],
            "晶粒粒径": [grain_size],
            "压溃压力": [pressure],
            "磨粒速度": [velocity]
        })
        
        try:
            # 准备数据
            input_data = prepare_material_data(
                temp_df,
                scaler=self.scaler,
                le=self.label_encoder,
                augment=False
            )
            
            # 转换为张量
            material_ids = torch.LongTensor(input_data["material_ids"]).to(self.device)
            numeric_raw = torch.FloatTensor(input_data["numeric_raw"]).to(self.device)
            
            # 预测
            with torch.no_grad():
                prediction, uncertainty, _, _ = self.model(material_ids, numeric_raw)
            
            # 转换为标量
            pred_value = float(prediction.cpu().squeeze().item())
            unc_value = float(np.sqrt(max(uncertainty.cpu().squeeze().item(), 1e-8)))
            
            return pred_value, unc_value
            
        except Exception as e:
            raise RuntimeError(f"单样本预测失败: {str(e)}")


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(
        description="材料裂纹深度预测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本预测 (使用默认路径)
  python model_pre.py --model ../results/improved_model.pth --train_data 3011916164精确材料性能数据.csv --input test.csv --output predictions.csv
  
  # 使用MC Dropout估计不确定度
  python model_pre.py --model ../results/improved_model.pth --train_data 3011916164精确材料性能数据.csv --input test.csv --output predictions.csv --mc_dropout --n_samples 20
  
  # 单样本预测
  python model_pre.py --model ../results/improved_model.pth --train_data 3011916164精确材料性能数据.csv --single --material 碳化硅 --hardness 9.5 --elastic_modulus 450 --fracture_toughness 4.5 --grain_size 0.00001 --pressure 8.5 --velocity 40
        """
    )
    
    # 修改为默认参数，与debug结果一致
    parser.add_argument("--model", default="../results/improved_model.pth", help="训练好的模型文件路径")
    parser.add_argument("--train_data", default="3011916164精确材料性能数据.csv", help="训练数据CSV文件路径")
    parser.add_argument("--input", help="输入数据CSV文件路径")
    parser.add_argument("--output", help="输出结果CSV文件路径")
    parser.add_argument("--mc_dropout", action="store_true", help="使用MC Dropout估计不确定度")
    parser.add_argument("--n_samples", type=int, default=10, help="MC Dropout采样次数")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="计算设备")
    
    # 单样本预测参数
    parser.add_argument("--single", action="store_true", help="单样本预测模式")
    parser.add_argument("--material", help="材料种类")
    parser.add_argument("--hardness", type=float, help="莫氏硬度")
    parser.add_argument("--elastic_modulus", type=float, help="弹性模量")
    parser.add_argument("--fracture_toughness", type=float, help="断裂韧性")
    parser.add_argument("--grain_size", type=float, help="晶粒粒径")
    parser.add_argument("--pressure", type=float, help="压溃压力")
    parser.add_argument("--velocity", type=float, help="磨粒速度")
    
    args = parser.parse_args()
    
    try:
        # 创建预测器
        predictor = CrackDepthPredictor(
            model_path=args.model,
            train_data_path=args.train_data,
            device=args.device
        )
        
        if args.single:
            # 单样本预测模式
            if not all([args.material, args.hardness is not None, args.elastic_modulus is not None, 
                       args.fracture_toughness is not None, args.grain_size is not None, 
                       args.pressure is not None, args.velocity is not None]):
                parser.error("单样本预测需要提供所有参数: --material, --hardness, --elastic_modulus, --fracture_toughness, --grain_size, --pressure, --velocity")
            
            print(f"\n{'='*60}")
            print("单样本预测")
            print(f"{'='*60}")
            print(f"材料种类: {args.material}")
            print(f"莫氏硬度: {args.hardness}")
            print(f"弹性模量: {args.elastic_modulus}")
            print(f"断裂韧性: {args.fracture_toughness}")
            print(f"晶粒粒径: {args.grain_size}")
            print(f"压溃压力: {args.pressure}")
            print(f"磨粒速度: {args.velocity}")
            
            pred, unc = predictor.predict_single(
                args.material, args.hardness, args.elastic_modulus,
                args.fracture_toughness, args.grain_size, args.pressure, args.velocity
            )
            
            print(f"\n预测结果:")
            print(f"  - 裂纹深度: {pred:.6f}")
            print(f"  - 不确定度: {unc:.6f}")
            print(f"  - 置信区间 (95%): [{pred - 1.96*unc:.6f}, {pred + 1.96*unc:.6f}]")
            print(f"{'='*60}")
            
        else:
            # 批量预测模式
            if not args.input:
                # 如果没有指定输入文件，使用默认的test.csv
                args.input = "test.csv"
                if not os.path.exists(args.input):
                    parser.error("未找到默认输入文件 test.csv，请使用 --input 指定输入文件")
            
            result_df = predictor.predict(
                input_data_path=args.input,
                output_path=args.output,
                use_mc_dropout=args.mc_dropout,
                n_samples=args.n_samples
            )
            
            # 显示前几行结果
            print("\n预测结果预览 (前5行):")
            print(result_df[["材料种类", "预测裂纹深度", "不确定度"]].head())
            
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())