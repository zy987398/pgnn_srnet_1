"""
符号解码器 (Symbolic Decoder)
============================
后处理模块：从训练好的PGNN模型中提取可解释的物理公式

功能：
1. 建立物理符号映射表 (Symbol Mapping)
2. 解析模型权重生成符号表达式
3. 使用SymPy进行数学简化和LaTeX渲染

使用方法：
    from symbolic_decoder import SymbolicDecoder
    
    decoder = SymbolicDecoder(model)
    formula = decoder.decode(material_ids, numeric_features)
    print(formula.latex)  # 输出LaTeX公式
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import sympy as sp
from sympy import symbols, sqrt, log, sin, tanh, exp, Rational, latex, simplify, nsimplify
from sympy import Float, Add, Mul, Pow


# ============================================================================
# 阶段一：物理符号映射表 (Symbol Mapping)
# ============================================================================

class PhysicsSymbolMapper:
    """
    物理符号映射器
    将特征索引映射到物理符号和LaTeX表示
    """
    
    def __init__(self):
        # 定义18个特征的符号映射
        # Feature 0-5: 原始物理特征
        # Feature 6-17: 物理衍生特征
        
        self.feature_symbols = {
            # === 原始特征 (0-5) ===
            0: {
                'name': 'Hardness',
                'symbol': 'H',
                'latex': r'H',
                'unit': 'GPa',
                'description': '莫氏硬度'
            },
            1: {
                'name': 'ElasticModulus', 
                'symbol': 'E',
                'latex': r'E',
                'unit': 'GPa',
                'description': '弹性模量'
            },
            2: {
                'name': 'FractureToughness',
                'symbol': 'K_c',
                'latex': r'K_c',
                'unit': 'MPa·√m',
                'description': '断裂韧性'
            },
            3: {
                'name': 'GrainSize',
                'symbol': 'd',
                'latex': r'd',
                'unit': 'μm',
                'description': '晶粒粒径'
            },
            4: {
                'name': 'Stress',
                'symbol': 'σ',
                'latex': r'\sigma',
                'unit': 'MPa',
                'description': '磨削压力'
            },
            5: {
                'name': 'Velocity',
                'symbol': 'v',
                'latex': r'v',
                'unit': 'm/s',
                'description': '磨削速度'
            },
            
            # === 物理衍生特征 (6-17) ===
            6: {
                'name': 'GriffithEnergyRelease',
                'symbol': 'G',
                'latex': r'G',
                'unit': 'J/m²',
                'description': 'Griffith能量释放率 G = σ²πd/E',
                'formula': r'\frac{\sigma^2 \pi d}{E}'
            },
            7: {
                'name': 'StressIntensityRatio',
                'symbol': 'K_ratio',
                'latex': r'\frac{K}{K_c}',
                'unit': '-',
                'description': '应力强度比 K/Kc = σ√(πd)/K_IC',
                'formula': r'\frac{\sigma \sqrt{\pi d}}{K_{IC}}'
            },
            8: {
                'name': 'HallPetch',
                'symbol': 'd_hp',
                'latex': r'd^{-1/2}',
                'unit': 'm^{-½}',
                'description': 'Hall-Petch尺度项',
                'formula': r'd^{-1/2}'
            },
            9: {
                'name': 'CrackShielding',
                'symbol': 'chi',
                'latex': r'\chi',
                'unit': 'm',
                'description': '裂纹屏蔽因子 χ = (K_IC/E)√d',
                'formula': r'\frac{K_{IC}}{E}\sqrt{d}'
            },
            10: {
                'name': 'LoadStrengthRatio',
                'symbol': 'rho_1',
                'latex': r'\rho_1',
                'unit': '-',
                'description': '载荷-强度比 ρ₁ = σ/H',
                'formula': r'\frac{\sigma}{H}'
            },
            11: {
                'name': 'PowerStrengthRatio',
                'symbol': 'rho_2',
                'latex': r'\rho_2',
                'unit': 'm/s',
                'description': '功率-强度比 ρ₂ = σV/H',
                'formula': r'\frac{\sigma V}{H}'
            },
            12: {
                'name': 'CriticalDepth',
                'symbol': 'd_c',
                'latex': r'd_c',
                'unit': 'm',
                'description': '延-脆转临界深度 d_c = η(E/H)(K_IC/H)²',
                'formula': r'\eta \frac{E}{H}\left(\frac{K_{IC}}{H}\right)^2'
            },
            13: {
                'name': 'CriticalCrackLengthRatio',
                'symbol': 'a_c_ratio',
                'latex': r'\frac{a_c}{d}',
                'unit': '-',
                'description': '临界裂纹长度比 a_c/d',
                'formula': r'\frac{1}{\pi d}\left(\frac{K_{IC}}{\sigma}\right)^2'
            },
            14: {
                'name': 'PlasticZoneSizeRatio',
                'symbol': 'r_p_ratio',
                'latex': r'\frac{r_p}{d}',
                'unit': '-',
                'description': '塑性区尺寸比 r_p/d',
                'formula': r'\frac{1}{2\pi d}\left(\frac{K_{IC}}{H}\right)^2'
            },
            15: {
                'name': 'BrittlenessIndex',
                'symbol': 'B',
                'latex': r'B',
                'unit': '√m⁻¹',
                'description': '脆性指数 B = H/K_IC',
                'formula': r'\frac{H}{K_{IC}}'
            },
            16: {
                'name': 'InitialProfileRatio',
                'symbol': 'K_E_ratio',
                'latex': r'\frac{K_{IC}}{E}',
                'unit': '√m',
                'description': '初-剖比 K_IC/E',
                'formula': r'\frac{K_{IC}}{E}'
            },
            17: {
                'name': 'ElasticStrainEnergyDensity',
                'symbol': 'U',
                'latex': r'U',
                'unit': 'J/m³',
                'description': '弹性应变能密度 U = 0.5σ²/E',
                'formula': r'\frac{\sigma^2}{2E}'
            }
        }
        
        # 算子映射
        self.operator_symbols = {
            0: {'name': 'identity', 'latex': r'x', 'sympy': lambda x: x},
            1: {'name': 'reciprocal', 'latex': r'\frac{1}{x}', 'sympy': lambda x: 1/x},
            2: {'name': 'square', 'latex': r'x^2', 'sympy': lambda x: x**2},
            3: {'name': 'cube', 'latex': r'x^3', 'sympy': lambda x: x**3},
            4: {'name': 'sqrt', 'latex': r'\sqrt{x}', 'sympy': lambda x: sp.sqrt(x)},
            5: {'name': 'log', 'latex': r'\log(x)', 'sympy': lambda x: sp.log(x)},
            6: {'name': 'tanh', 'latex': r'\tanh(x)', 'sympy': lambda x: sp.tanh(x)},
            7: {'name': 'sin', 'latex': r'\sin(x)', 'sympy': lambda x: sp.sin(x)}
        }
        
        # 创建SymPy符号
        self._create_sympy_symbols()
    
    def _create_sympy_symbols(self):
        """创建SymPy符号变量"""
        self.sympy_symbols = {}
        
        # 基础物理量符号
        self.H = sp.Symbol('H', positive=True, real=True)  # 硬度
        self.E = sp.Symbol('E', positive=True, real=True)  # 弹性模量
        self.K_c = sp.Symbol('K_c', positive=True, real=True)  # 断裂韧性
        self.d = sp.Symbol('d', positive=True, real=True)  # 晶粒粒径
        self.sigma = sp.Symbol('sigma', positive=True, real=True)  # 应力
        self.v = sp.Symbol('v', positive=True, real=True)  # 速度
        
        # 映射到字典
        self.sympy_symbols = {
            0: self.H,
            1: self.E,
            2: self.K_c,
            3: self.d,
            4: self.sigma,
            5: self.v,
            # 衍生特征用组合表达式表示
            6: self.sigma**2 * sp.pi * self.d / self.E,  # G
            7: self.sigma * sp.sqrt(sp.pi * self.d) / self.K_c,  # K/Kc
            8: 1 / sp.sqrt(self.d),  # Hall-Petch
            9: self.K_c / self.E * sp.sqrt(self.d),  # χ
            10: self.sigma / self.H,  # ρ1
            11: self.sigma * self.v / self.H,  # ρ2
            12: sp.Rational(15, 100) * self.E / self.H * (self.K_c / self.H)**2,  # d_c
            13: (self.K_c / self.sigma)**2 / (sp.pi * self.d),  # a_c/d
            14: (self.K_c / self.H)**2 / (2 * sp.pi * self.d),  # r_p/d
            15: self.H / self.K_c,  # B
            16: self.K_c / self.E,  # K_c/E
            17: self.sigma**2 / (2 * self.E)  # U
        }
        
        # 聚合特征符号（用于多项式）
        self.X_poly = sp.Symbol('X_{poly}', real=True)
    
    def get_feature_latex(self, idx: int) -> str:
        """获取特征的LaTeX表示"""
        if idx in self.feature_symbols:
            return self.feature_symbols[idx]['latex']
        return f'f_{{{idx}}}'
    
    def get_feature_sympy(self, idx: int) -> sp.Expr:
        """获取特征的SymPy表达式"""
        if idx in self.sympy_symbols:
            return self.sympy_symbols[idx]
        return sp.Symbol(f'f_{idx}')
    
    def get_operator_latex(self, idx: int, var: str = 'x') -> str:
        """获取算子的LaTeX表示"""
        if idx in self.operator_symbols:
            return self.operator_symbols[idx]['latex'].replace('x', var)
        return var
    
    def get_operator_sympy(self, idx: int) -> callable:
        """获取算子的SymPy函数"""
        if idx in self.operator_symbols:
            return self.operator_symbols[idx]['sympy']
        return lambda x: x


# ============================================================================
# 阶段二：符号解码器 (Symbolic Decoder)
# ============================================================================

@dataclass
class DecodedFormula:
    """解码后的公式数据结构"""
    # 原始组件
    feature_aggregation: str  # 特征聚合表达式
    polynomial_terms: str     # 多项式项
    operator_term: str        # 算子项
    gate_value: float         # 门控值
    
    # 完整公式
    full_formula: str         # 完整公式（字符串）
    latex: str                # LaTeX格式
    sympy_expr: sp.Expr       # SymPy表达式
    simplified_latex: str     # 简化后的LaTeX
    
    # 解释信息
    dominant_features: List[Tuple[str, float]]  # 主导特征及权重
    dominant_operator: str    # 主导算子
    interpretation: str       # 物理解释
    
    # 展开后的表达式（基于原始物理特征）- 带默认值的字段必须在最后
    expanded_polynomial_latex: str = ""  # 多项式项展开后的LaTeX
    expanded_operator_latex: str = ""    # 算子项展开后的LaTeX


class SymbolicDecoder:
    """
    符号解码器
    从训练好的PGNN模型中提取可解释的物理公式
    """
    
    def __init__(self, model, 
                 feature_threshold: float = 0.1,
                 coeff_threshold: float = 1e-3,
                 simplify_coeffs: bool = True,
                 decimal_places: int = 3,
                 use_raw_features: bool = True,
                 expand_formula: bool = True):
        """
        初始化解码器
        
        Args:
            model: 训练好的MaterialCrackPGNN模型
            feature_threshold: 特征权重阈值，低于此值的特征将被忽略
            coeff_threshold: 系数阈值，低于此值的项将被视为0
            simplify_coeffs: 是否简化系数（转为分数等）
            decimal_places: 保留的小数位数
            use_raw_features: 是否使用原始物理特征（True）或聚合符号X_poly（False）
            expand_formula: 是否展开多项式（True会展开(a+b)^2等形式）
        """
        self.model = model
        self.mapper = PhysicsSymbolMapper()
        self.feature_threshold = feature_threshold
        self.coeff_threshold = coeff_threshold
        self.simplify_coeffs = simplify_coeffs
        self.decimal_places = decimal_places
        self.use_raw_features = use_raw_features
        self.expand_formula = expand_formula
        
        # 缓存解释信息
        self._last_interpretation = None
        
    def decode(self, material_ids: torch.Tensor, 
               numeric_features: torch.Tensor,
               use_mean: bool = True) -> DecodedFormula:
        """
        解码模型输出为物理公式
        
        Args:
            material_ids: 材料ID张量
            numeric_features: 数值特征张量
            use_mean: 是否使用批次平均值（True）或取第一个样本（False）
            
        Returns:
            DecodedFormula: 解码后的公式对象
        """
        # 确保模型在评估模式
        self.model.eval()
        
        # 前向传播获取中间结果
        with torch.no_grad():
            _ = self.model(material_ids, numeric_features)
        
        # 获取符号层的解释信息
        interpretation = self.model.get_symbolic_interpretation()
        self._last_interpretation = interpretation
        
        # 提取权重
        var_weights = interpretation.get('variable_weights')
        op_weights = interpretation.get('operator_weights')
        coeffs = interpretation.get('coefficients')
        
        # 获取门控值
        gate_stats = self.model.get_gate_statistics()
        gate_value = gate_stats['mean']
        
        # 处理权重（取平均或第一个样本）
        if var_weights is not None:
            if use_mean:
                var_weights = var_weights.mean(dim=0).numpy()
            else:
                var_weights = var_weights[0].numpy()
        else:
            var_weights = np.ones(18) / 18
            
        if op_weights is not None:
            if use_mean:
                op_weights = op_weights.mean(dim=0).numpy()
            else:
                op_weights = op_weights[0].numpy()
        else:
            op_weights = np.ones(8) / 8
            
        if coeffs is not None:
            if use_mean:
                coeffs = coeffs.mean(dim=0).numpy()
            else:
                coeffs = coeffs[0].numpy()
        else:
            coeffs = np.zeros(4)
        
        # === 阶段2.1：特征聚合 ===
        feature_agg, dominant_features, X_poly_expr = self._decode_feature_aggregation(var_weights)
        
        # === 阶段2.2：多项式项解析 ===
        # 先用Φ符号解析（显示用）
        poly_terms_phi, _ = self._decode_polynomial(coeffs, X_poly_expr, use_raw_features=False)
        # 再用原始特征展开
        poly_terms, poly_expr = self._decode_polynomial(coeffs, X_poly_expr, use_raw_features=True)
        expanded_poly_latex = self._to_latex(poly_expr) if poly_expr != sp.Float(0) else ""
        
        # === 阶段2.3：算子项解析 ===
        # 先用Φ符号解析（显示用）
        op_term_phi, dominant_op, _ = self._decode_operators(op_weights, X_poly_expr, use_raw_features=False)
        # 再用原始特征展开
        op_term, _, op_expr = self._decode_operators(op_weights, X_poly_expr, use_raw_features=True)
        expanded_op_latex = self._to_latex(op_expr) if op_expr != sp.Float(0) else ""
        
        # === 阶段2.4：门控融合 ===
        full_formula, full_expr = self._apply_gating(
            gate_value, poly_expr, op_expr
        )
        
        # === 阶段3：SymPy简化 ===
        simplified_expr = self._simplify_expression(full_expr)
        simplified_latex = self._to_latex(simplified_expr)
        
        # 生成物理解释
        interpretation_text = self._generate_interpretation(
            dominant_features, dominant_op, gate_value, coeffs
        )
        
        return DecodedFormula(
            feature_aggregation=feature_agg,
            polynomial_terms=poly_terms_phi,  # 用Φ符号的表达式
            operator_term=op_term_phi,         # 用Φ符号的表达式
            gate_value=gate_value,
            full_formula=full_formula,
            latex=self._to_latex(full_expr),
            sympy_expr=full_expr,
            simplified_latex=simplified_latex,
            expanded_polynomial_latex=expanded_poly_latex,  # 展开后的多项式
            expanded_operator_latex=expanded_op_latex,       # 展开后的算子
            dominant_features=dominant_features,
            dominant_operator=dominant_op,
            interpretation=interpretation_text
        )
    
    def _decode_feature_aggregation(self, var_weights: np.ndarray) -> Tuple[str, List[Tuple[str, float]], sp.Expr]:
        """
        解码特征聚合
        
        Returns:
            (latex_str, dominant_features_list, sympy_expr)
        """
        # 筛选显著特征
        significant_indices = np.where(var_weights > self.feature_threshold)[0]
        
        if len(significant_indices) == 0:
            # 如果没有显著特征，取top-3
            significant_indices = np.argsort(var_weights)[-3:]
        
        # 构建表达式
        terms = []
        dominant_features = []
        sympy_terms = []
        
        for idx in significant_indices:
            weight = var_weights[idx]
            feature_latex = self.mapper.get_feature_latex(idx)
            feature_sympy = self.mapper.get_feature_sympy(idx)
            
            # 格式化权重
            weight_str = self._format_coefficient(weight)
            
            terms.append(f"{weight_str} \\cdot {feature_latex}")
            dominant_features.append((self.mapper.feature_symbols[idx]['name'], float(weight)))
            
            # SymPy表达式
            sympy_terms.append(sp.Float(weight, self.decimal_places) * feature_sympy)
        
        latex_str = " + ".join(terms)
        sympy_expr = sum(sympy_terms) if sympy_terms else sp.Float(0)
        
        # 排序dominant_features
        dominant_features.sort(key=lambda x: x[1], reverse=True)
        
        return latex_str, dominant_features, sympy_expr
    
    def _decode_polynomial(self, coeffs: np.ndarray, X_poly_expr: sp.Expr, use_raw_features: bool = True) -> Tuple[str, sp.Expr]:
        """
        解码多项式项
        coeffs: [c0, c1, c2, c3] 对应 [线性, 平方, 立方, 平方根]
        X_poly_expr: 实际的特征聚合SymPy表达式
        use_raw_features: 是否使用原始物理特征（True）或抽象符号Φ（False）
        """
        # 选择使用实际特征表达式还是抽象符号
        if use_raw_features:
            X = X_poly_expr  # 使用实际的特征聚合表达式
            X_latex = "\\Phi"  # 用Φ表示特征聚合（LaTeX显示用）
        else:
            X = self.mapper.X_poly  # 使用抽象符号
            X_latex = "\\Phi"  # 统一使用Φ符号
        
        terms = []
        sympy_terms = []
        
        # 线性项
        if abs(coeffs[0]) > self.coeff_threshold:
            c0_str = self._format_coefficient(coeffs[0])
            terms.append(f"{c0_str} \\cdot {X_latex}")
            sympy_terms.append(sp.Float(coeffs[0], self.decimal_places) * X)
        
        # 平方项
        if abs(coeffs[1]) > self.coeff_threshold:
            c1_str = self._format_coefficient(coeffs[1])
            terms.append(f"{c1_str} \\cdot {X_latex}^2")
            sympy_terms.append(sp.Float(coeffs[1], self.decimal_places) * X**2)
        
        # 立方项
        if abs(coeffs[2]) > self.coeff_threshold:
            c2_str = self._format_coefficient(coeffs[2])
            terms.append(f"{c2_str} \\cdot {X_latex}^3")
            sympy_terms.append(sp.Float(coeffs[2], self.decimal_places) * X**3)
        
        # 平方根项
        if abs(coeffs[3]) > self.coeff_threshold:
            c3_str = self._format_coefficient(coeffs[3])
            terms.append(f"{c3_str} \\cdot \\sqrt{{{X_latex}}}")
            # 对于实际特征表达式，使用更安全的sqrt处理
            if use_raw_features:
                sympy_terms.append(sp.Float(coeffs[3], self.decimal_places) * sp.sqrt(sp.Abs(X)))
            else:
                sympy_terms.append(sp.Float(coeffs[3], self.decimal_places) * sp.sqrt(sp.Abs(X)))
        
        if not terms:
            return "0", sp.Float(0)
        
        latex_str = " + ".join(terms)
        sympy_expr = sum(sympy_terms)
        
        return latex_str, sympy_expr
    
    def _decode_operators(self, op_weights: np.ndarray, X_poly_expr: sp.Expr, use_raw_features: bool = True) -> Tuple[str, str, sp.Expr]:
        """
        解码算子项
        找到权重最大的算子，或使用加权组合
        X_poly_expr: 实际的特征聚合SymPy表达式
        use_raw_features: 是否使用原始物理特征
        """
        # 选择使用实际特征表达式还是抽象符号
        if use_raw_features:
            X = X_poly_expr
            X_latex = "\\Phi"
        else:
            X = self.mapper.X_poly
            X_latex = "\\Phi"  # 统一使用Φ符号
        
        # 找到主导算子
        dominant_idx = np.argmax(op_weights)
        dominant_weight = op_weights[dominant_idx]
        dominant_op_name = self.mapper.operator_symbols[dominant_idx]['name']
        
        # 如果主导算子权重很高（>0.5），只使用它
        if dominant_weight > 0.5:
            op_latex = self.mapper.get_operator_latex(dominant_idx, X_latex)
            op_sympy_func = self.mapper.get_operator_sympy(dominant_idx)
            
            weight_str = self._format_coefficient(dominant_weight)
            latex_str = f"{weight_str} \\cdot {op_latex}"
            sympy_expr = sp.Float(dominant_weight, self.decimal_places) * op_sympy_func(X)
            
            return latex_str, dominant_op_name, sympy_expr
        
        # 否则，使用top-3算子的加权组合
        top_indices = np.argsort(op_weights)[-3:][::-1]
        
        terms = []
        sympy_terms = []
        
        for idx in top_indices:
            weight = op_weights[idx]
            if weight < self.coeff_threshold:
                continue
                
            op_latex = self.mapper.get_operator_latex(idx, X_latex)
            op_sympy_func = self.mapper.get_operator_sympy(idx)
            
            weight_str = self._format_coefficient(weight)
            terms.append(f"{weight_str} \\cdot {op_latex}")
            sympy_terms.append(sp.Float(weight, self.decimal_places) * op_sympy_func(X))
        
        latex_str = " + ".join(terms) if terms else "0"
        sympy_expr = sum(sympy_terms) if sympy_terms else sp.Float(0)
        
        return latex_str, dominant_op_name, sympy_expr
    
    def _apply_gating(self, gate: float, poly_expr: sp.Expr, op_expr: sp.Expr) -> Tuple[str, sp.Expr]:
        """
        应用门控融合
        Result = Gate * Polynomial + (1-Gate) * Operator
        """
        gate_str = self._format_coefficient(gate)
        one_minus_gate_str = self._format_coefficient(1 - gate)
        
        # LaTeX字符串
        latex_str = f"{gate_str} \\cdot T_{{poly}} + {one_minus_gate_str} \\cdot T_{{op}}"
        
        # SymPy表达式
        g = sp.Float(gate, self.decimal_places)
        sympy_expr = g * poly_expr + (1 - g) * op_expr
        
        return latex_str, sympy_expr
    
    def _simplify_expression(self, expr: sp.Expr) -> sp.Expr:
        """
        使用SymPy简化表达式
        """
        try:
            # 根据设置决定是否展开
            if self.expand_formula:
                # 完全展开多项式
                expanded = sp.expand(expr)
            else:
                # 保持因式形式，只做基本简化
                expanded = expr
            
            # 简化
            simplified = simplify(expanded)
            
            # 尝试将浮点数转为更好的表示
            if self.simplify_coeffs:
                # 用有理数近似小数
                simplified = nsimplify(simplified, tolerance=10**(-self.decimal_places))
            
            # 移除非常小的项（可选）
            simplified = self._remove_small_terms(simplified)
            
            return simplified
        except Exception as e:
            print(f"Warning: Simplification failed: {e}")
            return expr
    
    def _remove_small_terms(self, expr: sp.Expr, threshold: float = 1e-4) -> sp.Expr:
        """
        移除系数非常小的项
        """
        try:
            if expr.is_Add:
                # 如果是加法表达式，检查每一项
                new_terms = []
                for term in expr.args:
                    # 获取系数
                    coeff = term.as_coeff_Mul()[0]
                    if hasattr(coeff, 'evalf'):
                        coeff_val = abs(float(coeff.evalf()))
                    else:
                        coeff_val = abs(float(coeff)) if isinstance(coeff, (int, float)) else 1.0
                    
                    # 只保留系数较大的项
                    if coeff_val > threshold:
                        new_terms.append(term)
                
                if new_terms:
                    return sp.Add(*new_terms)
                else:
                    return sp.Float(0)
            return expr
        except:
            return expr
    
    def _to_latex(self, expr: sp.Expr) -> str:
        """
        转换为LaTeX格式
        """
        try:
            # 自定义LaTeX设置，使用 \\cdot 作为乘号
            latex_str = latex(expr, 
                             mul_symbol='\\cdot',
                             inv_trig_style='full')
            
            # 后处理：将sigma替换为\sigma
            latex_str = latex_str.replace('sigma', r'\sigma')
            
            return latex_str
        except Exception as e:
            print(f"Warning: LaTeX conversion failed: {e}")
            return str(expr)
    
    def _to_readable(self, expr: sp.Expr) -> str:
        """
        转换为易读的字符串格式（非LaTeX）
        """
        try:
            # 使用SymPy的字符串表示
            readable_str = str(expr)
            
            # 替换常见符号为更易读的形式
            readable_str = readable_str.replace('**', '^')
            readable_str = readable_str.replace('*', '·')
            readable_str = readable_str.replace('K_{IC}', 'K_IC')
            readable_str = readable_str.replace('sqrt', '√')
            readable_str = readable_str.replace('pi', 'π')
            
            return readable_str
        except Exception as e:
            print(f"Warning: Readable conversion failed: {e}")
            return str(expr)
    
    def _format_coefficient(self, value: float) -> str:
        """格式化系数"""
        if abs(value) < self.coeff_threshold:
            return "0"
        
        # 四舍五入
        rounded = round(value, self.decimal_places)
        
        # 移除多余的零
        if rounded == int(rounded):
            return str(int(rounded))
        
        return f"{rounded:.{self.decimal_places}f}".rstrip('0').rstrip('.')
    
    def _generate_interpretation(self, 
                                 dominant_features: List[Tuple[str, float]],
                                 dominant_op: str,
                                 gate_value: float,
                                 coeffs: np.ndarray) -> str:
        """
        生成物理解释文本
        """
        lines = []
        
        # 主导特征分析
        lines.append("【主导物理特征】")
        for name, weight in dominant_features[:3]:
            feature_info = None
            for idx, info in self.mapper.feature_symbols.items():
                if info['name'] == name:
                    feature_info = info
                    break
            if feature_info:
                lines.append(f"  - {feature_info['description']} ({feature_info['latex']}): 权重 {weight:.3f}")
        
        # 多项式分析
        lines.append("\n【多项式结构】")
        poly_terms = []
        if abs(coeffs[0]) > self.coeff_threshold:
            poly_terms.append(f"线性项(c={coeffs[0]:.3f})")
        if abs(coeffs[1]) > self.coeff_threshold:
            poly_terms.append(f"平方项(c={coeffs[1]:.3f})")
        if abs(coeffs[2]) > self.coeff_threshold:
            poly_terms.append(f"立方项(c={coeffs[2]:.3f})")
        if abs(coeffs[3]) > self.coeff_threshold:
            poly_terms.append(f"平方根项(c={coeffs[3]:.3f})")
        lines.append(f"  活跃项: {', '.join(poly_terms) if poly_terms else '无显著项'}")
        
        # 算子分析
        lines.append(f"\n【主导算子】")
        lines.append(f"  - {dominant_op}")
        
        # 门控分析
        lines.append(f"\n【融合机制】")
        lines.append(f"  - 门控值: {gate_value:.3f}")
        if gate_value > 0.7:
            lines.append("  - 解释: 模型强烈依赖多项式表达式（物理先验主导）")
        elif gate_value > 0.5:
            lines.append("  - 解释: 模型略偏向多项式表达式")
        elif gate_value > 0.3:
            lines.append("  - 解释: 模型在多项式和算子之间较好平衡")
        else:
            lines.append("  - 解释: 模型更依赖算子表达式")
        
        return "\n".join(lines)
    
    def get_full_report(self, formula: DecodedFormula) -> str:
        """
        生成完整的公式报告
        """
        report = []
        report.append("=" * 60)
        report.append("           符号回归公式提取报告")
        report.append("=" * 60)
        
        # 0. 符号说明
        report.append("\n【符号说明】")
        report.append("  H     = 莫氏硬度 (Hardness) [GPa]")
        report.append("  E     = 弹性模量 (Elastic Modulus) [GPa]")
        report.append("  K_c   = 断裂韧性 (Fracture Toughness) [MPa·√m]")
        report.append("  d     = 磨粒粒径 (Grain Size) [μm]")
        report.append("  σ     = 磨削压力 (Stress) [MPa]")
        report.append("  v     = 磨削速度 (Velocity) [m/s]")
        report.append("  π     = 圆周率 ≈ 3.14159")
        
        # 1. 特征聚合表达式
        report.append("\n【1. 特征聚合表达式 (Φ)】")
        report.append("  特征加权组合:")
        report.append(f"  $\\Phi = {formula.feature_aggregation}$")
        
        # 2. 主导物理特征
        report.append("\n【2. 主导物理特征】")
        for i, (name, weight) in enumerate(formula.dominant_features[:3]):
            feature_info = None
            for idx, info in self.mapper.feature_symbols.items():
                if info['name'] == name:
                    feature_info = info
                    break
            if feature_info:
                report.append(f"  {i+1}. {feature_info['latex']} ({feature_info['description']}): 权重 {weight:.4f}")
        
        # 3. 多项式表达式
        report.append("\n【3. 多项式表达式】")
        report.append(f"  $T_{{poly}} = {formula.polynomial_terms}$")
        
        # 4. 算子表达式
        report.append("\n【4. 算子表达式】")
        report.append(f"  $T_{{op}} = {formula.operator_term}$")
        report.append(f"  主导算子: {formula.dominant_operator}")
        
        # 5. 门控融合
        report.append(f"\n【5. 门控融合】(Gate = {formula.gate_value:.3f})")
        report.append(f"  公式形式: $h = {formula.gate_value:.3f} \\cdot T_{{poly}} + {1-formula.gate_value:.3f} \\cdot T_{{op}}$")
        
        # 6. 展开后的完整公式
        report.append("\n【6. 展开后的完整公式 (基于原始物理特征)】")
        report.append(f"  ${formula.simplified_latex}$")
        
        # 7. 多项式表达式展开（基于原始物理特征）
        report.append("\n【7. 多项式表达式展开 (基于原始物理特征)】")
        if hasattr(formula, 'expanded_polynomial_latex') and formula.expanded_polynomial_latex:
            report.append(f"  $T_{{poly}} = {formula.expanded_polynomial_latex}$")
        else:
            report.append(f"  （已融合到完整公式中）")
        
        # 8. 算子表达式展开（基于原始物理特征）
        report.append("\n【8. 算子表达式展开 (基于原始物理特征)】")
        if hasattr(formula, 'expanded_operator_latex') and formula.expanded_operator_latex:
            report.append(f"  $T_{{op}} = {formula.expanded_operator_latex}$")
        else:
            report.append(f"  （已融合到完整公式中）")
        
        # 分隔线
        report.append("\n" + "-" * 60)
        
        # 物理解释
        report.append(formula.interpretation)
        
        # LaTeX版本
        report.append("\n" + "-" * 60)
        report.append("【可复制的LaTeX公式】")
        report.append("\\begin{equation}")
        report.append(f"  h = {formula.simplified_latex}")
        report.append("\\end{equation}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def export_to_file(self, formula: DecodedFormula, filepath: str):
        """
        导出公式到文件
        """
        report = self.get_full_report(formula)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
            f.write("\n\n")
            f.write("# 可直接复制到LaTeX文档的公式:\n")
            f.write(f"\\begin{{equation}}\n")
            f.write(f"  h = {formula.simplified_latex}\n")
            f.write(f"\\end{{equation}}\n")
        
        print(f"公式报告已导出到: {filepath}")


# ============================================================================
# 便捷函数
# ============================================================================

def extract_formula(model, material_ids, numeric_features, 
                    feature_threshold=0.1, verbose=True) -> DecodedFormula:
    """
    从模型中提取公式的便捷函数
    
    Args:
        model: 训练好的MaterialCrackPGNN模型
        material_ids: 材料ID张量
        numeric_features: 数值特征张量
        feature_threshold: 特征权重阈值
        verbose: 是否打印详细信息
        
    Returns:
        DecodedFormula: 解码后的公式
    """
    decoder = SymbolicDecoder(model, feature_threshold=feature_threshold)
    formula = decoder.decode(material_ids, numeric_features)
    
    if verbose:
        print(decoder.get_full_report(formula))
    
    return formula


def batch_extract_formulas(model, dataloader, n_batches=5, 
                          feature_threshold=0.1) -> List[DecodedFormula]:
    """
    从多个批次中提取公式，分析稳定性
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        n_batches: 分析的批次数
        feature_threshold: 特征权重阈值
        
    Returns:
        List[DecodedFormula]: 公式列表
    """
    decoder = SymbolicDecoder(model, feature_threshold=feature_threshold)
    formulas = []
    
    for i, (material_ids, numeric_features, _) in enumerate(dataloader):
        if i >= n_batches:
            break
        
        formula = decoder.decode(material_ids, numeric_features)
        formulas.append(formula)
    
    return formulas


def analyze_formula_stability(formulas: List[DecodedFormula]) -> Dict:
    """
    分析多个公式的稳定性
    """
    if not formulas:
        return {}
    
    # 统计主导特征
    feature_counts = {}
    for f in formulas:
        for name, _ in f.dominant_features[:3]:
            feature_counts[name] = feature_counts.get(name, 0) + 1
    
    # 统计主导算子
    op_counts = {}
    for f in formulas:
        op_counts[f.dominant_operator] = op_counts.get(f.dominant_operator, 0) + 1
    
    # 门控值统计
    gate_values = [f.gate_value for f in formulas]
    
    return {
        'feature_frequency': feature_counts,
        'operator_frequency': op_counts,
        'gate_mean': np.mean(gate_values),
        'gate_std': np.std(gate_values),
        'n_samples': len(formulas)
    }


# ============================================================================
# 主程序示例
# ============================================================================

if __name__ == "__main__":
    print("符号解码器模块")
    print("=" * 40)
    print("使用方法:")
    print("  from symbolic_decoder import SymbolicDecoder, extract_formula")
    print("  ")
    print("  # 方法1: 使用便捷函数")
    print("  formula = extract_formula(model, material_ids, numeric_features)")
    print("  print(formula.simplified_latex)")
    print("  ")
    print("  # 方法2: 使用解码器类")
    print("  decoder = SymbolicDecoder(model)")
    print("  formula = decoder.decode(material_ids, numeric_features)")
    print("  decoder.export_to_file(formula, 'formula_report.txt')")
