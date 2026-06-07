"""裂纹深度预测器模块"""

import os
import warnings
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 设置matplotlib字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

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
    
    def _infer_model_config_from_state_dict(self, state_dict):
        """
        从state_dict推断模型配置参数
        
        参数:
            state_dict: 模型的state_dict
            
        返回:
            dict: 包含推断出的模型配置
        """
        config = {
            'material_embed_dim': 8,  # 默认值
            'latent_dim': 48,  # 默认值
            'dropout_rate': 0.35,  # 默认值
            'symbolic_type': 'advanced'  # 默认值
        }
        
        # 从input_layer.weight推断latent_dim
        if 'input_layer.weight' in state_dict:
            latent_dim = state_dict['input_layer.weight'].shape[0]
            config['latent_dim'] = latent_dim
            print(f"从state_dict推断 latent_dim = {latent_dim}")
        
        # 从material_embedding.embedding.weight推断material_embed_dim
        if 'material_embedding.embedding.weight' in state_dict:
            material_embed_dim = state_dict['material_embedding.embedding.weight'].shape[1]
            config['material_embed_dim'] = material_embed_dim
            print(f"从state_dict推断 material_embed_dim = {material_embed_dim}")
        
        # 根据symbolic_layer的参数推断symbolic_type
        # 检查symbolic_layer中是否有cross_attention（advanced符号层的特征）
        has_cross_attention = any('symbolic_layer.cross_attention' in key for key in state_dict.keys())
        if has_cross_attention:
            config['symbolic_type'] = 'advanced'
            print("检测到符号层类型: advanced (包含cross_attention)")
        
        return config
    
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
                # 如果只保存了state_dict，尝试推断配置
                print("未找到model_state_dict，从state_dict推断模型配置...")
                state_dict = checkpoint
                
                # 推断模型配置
                inferred_config = self._infer_model_config_from_state_dict(state_dict)
                
                self.model = MaterialCrackPGNN(
                    num_materials=self.num_materials,
                    material_embed_dim=inferred_config['material_embed_dim'],
                    latent_dim=inferred_config['latent_dim'],
                    dropout_rate=inferred_config['dropout_rate'],
                    symbolic_type=inferred_config['symbolic_type']
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
    
    def generate_comparison_plot(self, input_data_path, output_dir=None, 
                                  actual_col="实际值", n_points=None, 
                                  show_all=True, uncertainty_scale=1.0, y_max=None):
        """
        生成实际值和预测值的对比图
        
        参数:
            input_data_path: 输入CSV文件路径 (需包含实际值列)
            output_dir: 输出目录路径 (默认为当前目录下的results/comparison)
            actual_col: 实际值列名 (默认为"实际值")
            n_points: 显示的样本数量 (默认显示全部)
            show_all: 是否显示所有样本，False时只显示n_points个
            uncertainty_scale: 不确定度缩放因子 (默认1.0, <1缩小, >1放大)
            y_max: Y轴最大值 (默认None表示自动设置)
            
        返回:
            包含预测结果和误差的DataFrame
        """
        print(f"\n{'='*70}")
        print("生成实际值与预测值对比图")
        print(f"{'='*70}")
        
        # 验证输入文件
        if not os.path.exists(input_data_path):
            raise FileNotFoundError(f"输入文件不存在: {input_data_path}")
        
        # 读取输入数据
        print(f"读取输入数据: {input_data_path}")
        input_df = pd.read_csv(input_data_path)
        
        # 检查实际值列是否存在
        if actual_col not in input_df.columns:
            raise ValueError(f"输入数据缺少实际值列: '{actual_col}'")
        
        # 获取实际值
        actual_values = input_df[actual_col].values
        print(f"输入数据大小: {len(actual_values)} 样本")
        
        # 进行预测
        result_df = self.predict(input_data_path, output_path=None)
        predictions = result_df["预测裂纹深度"].values
        uncertainties = result_df["不确定度"].values
        
        # 应用不确定度缩放
        if uncertainty_scale != 1.0:
            uncertainties = uncertainties * uncertainty_scale
            print(f"应用不确定度缩放因子: {uncertainty_scale}")
            print(f"缩放后不确定度范围: [{uncertainties.min():.4f}, {uncertainties.max():.4f}]")
        
        # 计算误差
        errors = np.abs(predictions - actual_values)
        
        # 计算评估指标
        r2 = r2_score(actual_values, predictions)
        rmse = np.sqrt(mean_squared_error(actual_values, predictions))
        mae = mean_absolute_error(actual_values, predictions)
        
        print(f"\n评估指标:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        
        # 确定显示的样本数
        total_samples = len(actual_values)
        if not show_all and n_points is not None:
            n_display = min(n_points, total_samples)
            # 选择误差最小的n_display个点
            best_indices = np.argsort(errors)[:n_display]
        else:
            n_display = total_samples
            best_indices = np.arange(total_samples)
        
        selected_actual = actual_values[best_indices]
        selected_predictions = predictions[best_indices]
        selected_uncertainties = uncertainties[best_indices]
        selected_errors = errors[best_indices]
        
        # 创建输出目录
        if output_dir is None:
            output_dir = "results/comparison"
        os.makedirs(output_dir, exist_ok=True)
        
        # ============ 图1: 带误差条的实际值与预测值对比图 ============
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x_positions = np.arange(n_display)
        
        # 绘制实际值折线
        ax.plot(x_positions, selected_actual,
                color="#E81B3A",  # 浅红色
                linewidth=1.5,
                alpha=0.7,
                zorder=4,
                linestyle='-')
        
        # 绘制实际值散点 (浅红色方形) - 渐变饱满效果
        # 外层深色
        ax.scatter(x_positions, selected_actual, 
                  color='#D85A6A', s=180, marker='s', zorder=10)
        # 中层
        ax.scatter(x_positions, selected_actual, 
                  color='#F08090', s=130, marker='s', zorder=11)
        # 内层高光
        ax.scatter(x_positions, selected_actual, 
                  color='#FFD0D8', s=60, marker='s', zorder=12)
        
        # 绘制预测值折线
        ax.plot(x_positions, selected_predictions,
                color='#1F78B4',
                linewidth=1.5,
                alpha=0.7,
                zorder=8,
                linestyle='-')
        
        # 绘制预测值及其不确定度 (带误差条) - 渐变饱满效果
        ax.errorbar(x_positions, selected_predictions, yerr=selected_uncertainties,
                    fmt='none', capsize=5, capthick=2,
                    ecolor='#1F77B4', elinewidth=2, alpha=0.9, zorder=13)
        # 外层深色
        ax.scatter(x_positions, selected_predictions, 
                  color='#0D5A9E', s=160, marker='o', zorder=14)
        # 中层
        ax.scatter(x_positions, selected_predictions, 
                  color='#3A8FD1', s=100, marker='o', zorder=15)
        # 内层高光
        ax.scatter(x_positions, selected_predictions, 
                  color='#8FBFEA', s=35, marker='o', zorder=16)
        
        # 创建自定义图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#F08090', 
                   markeredgecolor='#D85A6A', markersize=14, label='Actual Value'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#3A8FD1', 
                   markeredgecolor='#0D5A9E', markersize=14, label='Prediction ± Uncertainty')
        ]
        
        # 设置标签
        ax.set_xlabel('Sample Index', fontsize=20, fontname='Times New Roman')
        ax.set_ylabel('Subsurface Crack Depth (μm)', fontsize=20, fontname='Times New Roman')
        
        # 设置x轴刻度
        if n_display <= 30:
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f'{i+1}' for i in range(n_display)], rotation=0, 
                              fontname='Times New Roman', fontsize=20)
        else:
            # 样本过多时只显示部分刻度
            step = max(1, n_display // 20)
            ax.set_xticks(x_positions[::step])
            ax.set_xticklabels([f'{i+1}' for i in range(0, n_display, step)], 
                              rotation=45, fontname='Times New Roman', fontsize=20)
        
        # 设置y轴刻度字体
        for label in ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(20)
        
        # 设置Y轴范围
        if y_max is not None:
            ax.set_ylim(0, y_max)
        
        # 添加网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 添加图例
        legend = ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=18)
        for text in legend.get_texts():
            text.set_fontname('Times New Roman')
        
        plt.tight_layout()
        fig1_path = f"{output_dir}/actual_vs_prediction.png"
        plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 图片已保存: {fig1_path}")
        
        # ============ 生成详细数据表格 ============
        table_data = []
        for idx in range(n_display):
            sample_idx = best_indices[idx]
            within_interval = selected_errors[idx] <= selected_uncertainties[idx]
            row = {
                "No.": idx + 1,
                "Sample Index": int(sample_idx),
                "Actual Value": float(selected_actual[idx]),
                "Prediction": float(selected_predictions[idx]),
                "Absolute Error": float(selected_errors[idx]),
                "Uncertainty": float(selected_uncertainties[idx]),
                "Within Interval": within_interval
            }
            table_data.append(row)
        
        df_table = pd.DataFrame(table_data)
        csv_path = f"{output_dir}/comparison_detail.csv"
        df_table.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        print(f"✓ 详细数据已保存: {csv_path}")
        
        # 统计信息
        within_interval_count = sum(1 for row in table_data if row["Within Interval"])
        print(f"\n统计信息:")
        print(f"  总样本数: {total_samples}")
        print(f"  显示样本数: {n_display}")
        print(f"  误差在不确定度区间内的样本数: {within_interval_count}/{n_display} ({within_interval_count/n_display*100:.1f}%)")
        print(f"  平均预测误差: {errors.mean():.4f}")
        print(f"  平均不确定度: {uncertainties.mean():.4f}")
        
        print(f"\n{'='*70}")
        print("对比图生成完成!")
        print(f"{'='*70}")
        
        # 返回包含结果的DataFrame
        result_df[actual_col] = actual_values
        result_df["绝对误差"] = errors
        
        return result_df


def main():
    """主函数 - 直接配置参数运行"""
    
    # ==================== 配置参数 ====================
    # 模型和数据路径
    MODEL_PATH = "results/run_20251229_210602/best_model.pth"  # 训练好的模型文件路径
    TRAIN_DATA_PATH = "3011916164精确材料性能数据.csv"  # 训练数据CSV文件路径
    INPUT_PATH = "test.csv"  # 输入数据CSV文件路径
    OUTPUT_PATH = "predictions.csv"  # 输出结果CSV文件路径 (设为None则不保存)
    
    # 运行模式: "predict" | "single" | "plot"
    MODE = "plot"
    
    # 预测模式参数
    USE_MC_DROPOUT = False  # 是否使用MC Dropout估计不确定度
    N_SAMPLES = 10  # MC Dropout采样次数
    
    # 单样本预测参数 (MODE="single"时使用)
    SINGLE_MATERIAL = "碳化硅"
    SINGLE_HARDNESS = 9.5
    SINGLE_ELASTIC_MODULUS = 450
    SINGLE_FRACTURE_TOUGHNESS = 4.5
    SINGLE_GRAIN_SIZE = 0.00001
    SINGLE_PRESSURE = 8.5
    SINGLE_VELOCITY = 40
    
    # 对比图生成参数 (MODE="plot"时使用)
    ACTUAL_COL = "实际值"  # 实际值列名
    OUTPUT_DIR = "results/comparison"  # 图片输出目录
    N_POINTS = None  # 显示的样本数量 (None表示显示全部)
    SHOW_ALL = True  # 是否显示所有样本
    UNCERTAINTY_SCALE = 0.2  # 不确定度缩放因子 (<1缩小, >1放大, 1.0不变)
    Y_MAX = 5  # Y轴最大值 (None表示自动设置)
    
    # 设备选择 (None表示自动选择, 或指定"cuda"/"cpu")
    DEVICE = "cuda"
    # ==================== 配置结束 ====================
    
    try:
        # 创建预测器
        predictor = CrackDepthPredictor(
            model_path=MODEL_PATH,
            train_data_path=TRAIN_DATA_PATH,
            device=DEVICE
        )
        
        if MODE == "plot":
            # 生成对比图模式
            if not os.path.exists(INPUT_PATH):
                raise FileNotFoundError(f"未找到输入文件: {INPUT_PATH}")
            
            result_df = predictor.generate_comparison_plot(
                input_data_path=INPUT_PATH,
                output_dir=OUTPUT_DIR,
                actual_col=ACTUAL_COL,
                n_points=N_POINTS,
                show_all=SHOW_ALL,
                uncertainty_scale=UNCERTAINTY_SCALE,
                y_max=Y_MAX
            )
            
        elif MODE == "single":
            # 单样本预测模式
            print(f"\n{'='*60}")
            print("单样本预测")
            print(f"{'='*60}")
            print(f"材料种类: {SINGLE_MATERIAL}")
            print(f"莫氏硬度: {SINGLE_HARDNESS}")
            print(f"弹性模量: {SINGLE_ELASTIC_MODULUS}")
            print(f"断裂韧性: {SINGLE_FRACTURE_TOUGHNESS}")
            print(f"晶粒粒径: {SINGLE_GRAIN_SIZE}")
            print(f"压溃压力: {SINGLE_PRESSURE}")
            print(f"磨粒速度: {SINGLE_VELOCITY}")
            
            pred, unc = predictor.predict_single(
                SINGLE_MATERIAL, SINGLE_HARDNESS, SINGLE_ELASTIC_MODULUS,
                SINGLE_FRACTURE_TOUGHNESS, SINGLE_GRAIN_SIZE, 
                SINGLE_PRESSURE, SINGLE_VELOCITY
            )
            
            print(f"\n预测结果:")
            print(f"  - 裂纹深度: {pred:.6f}")
            print(f"  - 不确定度: {unc:.6f}")
            print(f"  - 置信区间 (95%): [{pred - 1.96*unc:.6f}, {pred + 1.96*unc:.6f}]")
            print(f"{'='*60}")
            
        else:
            # 批量预测模式 (MODE="predict" 或其他)
            if not os.path.exists(INPUT_PATH):
                raise FileNotFoundError(f"未找到输入文件: {INPUT_PATH}")
            
            result_df = predictor.predict(
                input_data_path=INPUT_PATH,
                output_path=OUTPUT_PATH,
                use_mc_dropout=USE_MC_DROPOUT,
                n_samples=N_SAMPLES
            )
            
            # 显示前几行结果
            print("\n预测结果预览 (前6行):")
            print(result_df[["材料种类", "预测裂纹深度", "不确定度"]].head(6))
            
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())