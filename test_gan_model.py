#!/usr/bin/env python3
"""
SecureRouteX GAN Model - Quick Test
==================================
"""

from secureroutex_gan_model import SecureRouteXGAN
import numpy as np

def quick_test():
    print("🧪 SecureRouteX GAN Model - Quick Test")
    print("="*40)
    
    # Initialize model
    secureroutex = SecureRouteXGAN(input_dim=50, latent_dim=100)
    
    # Build architecture
    print("\n🔧 Building Model Components...")
    secureroutex.build_generator()
    secureroutex.build_discriminator() 
    secureroutex.build_trust_evaluator()
    secureroutex.build_attack_detector()
    
    print("\n✅ All Models Built Successfully!")
    
    # Test synthetic generation (without training)
    print("\n🎯 Testing Synthetic Attack Generation...")
    
    # Create dummy noise and conditions for testing
    noise = np.random.normal(0, 1, (5, 100))
    domain_conditions = np.array([[1,0,0], [0,1,0], [0,0,1], [1,0,0], [0,1,0]])  # 3 domains
    attack_conditions = np.array([[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]])  # 5 attack types
    
    # Generate synthetic data
    synthetic_data = secureroutex.generator.predict([noise, domain_conditions, attack_conditions])
    
    print(f"   • Generated Shape: {synthetic_data.shape}")
    print(f"   • Sample Values: {synthetic_data[0][:5]}")
    
    # Test trust evaluation
    print("\n🛡️ Testing Trust Evaluation...")
    dummy_network_data = np.random.uniform(-1, 1, (3, 50))
    dummy_domain_context = np.array([[1,0,0], [0,1,0], [0,0,1]])
    
    trust_outputs = secureroutex.trust_evaluator.predict([dummy_network_data, dummy_domain_context])
    
    print(f"   • Direct Trust: {trust_outputs[0][:3].flatten()}")
    print(f"   • Indirect Trust: {trust_outputs[1][:3].flatten()}")
    print(f"   • Energy Trust: {trust_outputs[2][:3].flatten()}")
    print(f"   • Composite Trust: {trust_outputs[3][:3].flatten()}")
    
    print("\n✅ Quick Test Complete - All Components Working!")
    print("🚀 Ready for full training with your dataset!")

if __name__ == "__main__":
    quick_test()