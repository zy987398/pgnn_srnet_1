# test_step2_fix.py
# 验证第二步修复是否成功

import torch
import numpy as np
from improved_material_crack_pgnn import MaterialCrackPGNN, PhysicsInformedFeatureExtractor

def test_physics_extractor():
    """测试物理特征提取器的数值稳定性"""
    print("Testing PhysicsInformedFeatureExtractor...")
    
    extractor = PhysicsInformedFeatureExtractor()
    
    # 创建包含潜在问题值的测试数据
    test_cases = [
        # 正常值
        torch.tensor([[5.0, 200.0, 3.0, 50.0, 1.0, 10.0]]),
        # 包含极小值
        torch.tensor([[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]]),
        # 包含极大值
        torch.tensor([[100.0, 1e6, 100.0, 1000.0, 100.0, 1000.0]]),
        # 包含负值
        torch.tensor([[-1.0, -100.0, -1.0, -10.0, -5.0, -50.0]]),
        # 混合极端值
        torch.tensor([[1e-8, 1e8, 1e-6, 1e6, 0.0, 1e-3]]),
    ]
    
    all_passed = True
    
    for i, test_input in enumerate(test_cases):
        try:
            output = extractor(test_input)
            
            # 检查输出是否包含NaN或Inf
            if torch.isnan(output).any():
                print(f"❌ Test case {i+1}: Contains NaN")
                all_passed = False
            elif torch.isinf(output).any():
                print(f"❌ Test case {i+1}: Contains Inf")
                all_passed = False
            else:
                print(f"✅ Test case {i+1}: Passed (output shape: {output.shape})")
                
        except Exception as e:
            print(f"❌ Test case {i+1}: Exception - {e}")
            all_passed = False
    
    return all_passed

def test_model_forward():
    """测试完整模型的前向传播"""
    print("\nTesting full model forward pass...")
    
    model = MaterialCrackPGNN(num_materials=3, latent_dim=32)
    model.eval()
    
    # 创建测试输入
    material_ids = torch.LongTensor([0, 1, 2, 0, 1])
    
    # 测试多种极端情况的特征
    test_features = [
        torch.randn(5, 6),  # 正常随机值
        torch.randn(5, 6) * 1e-6,  # 极小值
        torch.randn(5, 6) * 1e3,   # 大值
        torch.zeros(5, 6),         # 零值
    ]
    
    all_passed = True
    
    for i, features in enumerate(test_features):
        try:
            with torch.no_grad():
                pred, uncertainty, symbolic, residual = model(material_ids, features)
                
                # 检查所有输出
                outputs = {"pred": pred, "uncertainty": uncertainty, 
                          "symbolic": symbolic, "residual": residual}
                
                for name, output in outputs.items():
                    if torch.isnan(output).any():
                        print(f"❌ Test {i+1}: {name} contains NaN")
                        all_passed = False
                    elif torch.isinf(output).any():
                        print(f"❌ Test {i+1}: {name} contains Inf")
                        all_passed = False
                
                if all_passed:
                    print(f"✅ Test {i+1}: All outputs clean")
                    
        except Exception as e:
            print(f"❌ Test {i+1}: Exception - {e}")
            all_passed = False
    
    return all_passed

def test_gradient_flow():
    """测试梯度是否会爆炸"""
    print("\nTesting gradient flow...")
    
    model = MaterialCrackPGNN(num_materials=3, latent_dim=32)
    model.train()
    
    # 创建测试数据
    material_ids = torch.LongTensor([0, 1, 2])
    features = torch.randn(3, 6)
    target = torch.randn(3, 1)
    
    try:
        pred, uncertainty, _, _ = model(material_ids, features)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        
        # 检查梯度
        max_grad = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                max_grad = max(max_grad, grad_norm)
        
        if max_grad > 100:
            print(f"❌ Large gradient detected: {max_grad:.2f}")
            return False
        else:
            print(f"✅ Gradient flow normal (max: {max_grad:.2f})")
            return True
            
    except Exception as e:
        print(f"❌ Gradient test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("第二步修复验证测试")
    print("=" * 50)
    
    # 运行所有测试
    test1 = test_physics_extractor()
    test2 = test_model_forward()
    test3 = test_gradient_flow()
    
    print("\n" + "=" * 50)
    if test1 and test2 and test3:
        print("🎉 所有测试通过！第二步修复成功！")
        print("现在可以重新运行训练，应该会看到显著改善。")
    else:
        print("❌ 部分测试失败，请检查代码修改是否正确。")
    print("=" * 50)