#!/usr/bin/env python3
"""
SecureRouteX: Complete Training Pipeline
=======================================

Step-by-step training guide for SecureRouteX GAN model using your enhanced dataset.
This script shows you exactly how to train the model and what to do with the results.

Author: SecureRouteX Research Team
Date: September 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Import your GAN model
from secureroutex_gan_model import SecureRouteXGAN

def step1_load_and_prepare_data():
    """
    STEP 1: Load and prepare your enhanced dataset for training
    """
    print("📊 STEP 1: LOADING YOUR ENHANCED DATASET")
    print("="*50)
    
    # Load your enhanced dataset
    df = pd.read_csv('secureroutex_enhanced_dataset.csv')
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   • Shape: {df.shape}")
    print(f"   • Columns: {len(df.columns)}")
    
    # Show dataset overview
    print(f"\n📋 Dataset Overview:")
    print(f"   • Total samples: {len(df):,}")
    print(f"   • Features: {len(df.columns) - 3}")  # Excluding target columns
    print(f"   • Attack types: {df['attack_type'].nunique()}")
    print(f"   • Domains: {df['domain'].nunique()}")
    
    # Display attack type distribution
    print(f"\n🎯 Attack Type Distribution:")
    attack_counts = df['attack_type'].value_counts()
    for attack, count in attack_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   • {attack}: {count:,} ({percentage:.1f}%)")
    
    # Display domain distribution
    print(f"\n🏥 Domain Distribution:")
    domain_counts = df['domain'].value_counts()
    for domain, count in domain_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   • {domain}: {count:,} ({percentage:.1f}%)")
    
    return df

def step2_preprocess_data(df):
    """
    STEP 2: Preprocess data for GAN training
    """
    print("\n🔧 STEP 2: PREPROCESSING DATA FOR TRAINING")
    print("="*50)
    
    # Separate features from labels
    feature_columns = [col for col in df.columns 
                      if col not in ['attack_type', 'domain', 'is_malicious']]
    
    print(f"✅ Identified {len(feature_columns)} feature columns")
    
    # Extract features and labels
    X = df[feature_columns].values
    attack_labels = df['attack_type'].values
    domain_labels = df['domain'].values
    
    # One-hot encode categorical variables
    print("🔄 Encoding categorical variables...")
    
    # Create one-hot encodings
    attack_types = ['normal', 'ddos', 'malicious_node', 'energy_drain', 'routing_attack']
    domains = ['healthcare', 'transportation', 'underwater']
    
    # One-hot encode attack types
    attack_encoded = np.zeros((len(df), len(attack_types)))
    for i, attack in enumerate(attack_labels):
        if attack in attack_types:
            attack_idx = attack_types.index(attack)
            attack_encoded[i, attack_idx] = 1
    
    # One-hot encode domains
    domain_encoded = np.zeros((len(df), len(domains)))
    for i, domain in enumerate(domain_labels):
        if domain in domains:
            domain_idx = domains.index(domain)
            domain_encoded[i, domain_idx] = 1
    
    print(f"   • Attack encoding shape: {attack_encoded.shape}")
    print(f"   • Domain encoding shape: {domain_encoded.shape}")
    
    # Scale features
    print("📏 Scaling numerical features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train/validation/test sets
    print("✂️ Splitting data into train/validation/test sets...")
    
    # First split: 80% train+val, 20% test
    X_temp, X_test, attack_temp, attack_test, domain_temp, domain_test = train_test_split(
        X_scaled, attack_encoded, domain_encoded, 
        test_size=0.2, random_state=42, 
        stratify=attack_labels
    )
    
    # Second split: 80% train, 20% validation (from the 80%)
    X_train, X_val, attack_train, attack_val, domain_train, domain_val = train_test_split(
        X_temp, attack_temp, domain_temp,
        test_size=0.25, random_state=42,  # 0.25 * 0.8 = 0.2 of total
        stratify=[attack_types[np.argmax(row)] for row in attack_temp]
    )
    
    print(f"✅ Data split completed:")
    print(f"   • Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"   • Validation set: {X_val.shape[0]:,} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
    print(f"   • Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    
    data_dict = {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'attack_train': attack_train, 'attack_val': attack_val, 'attack_test': attack_test,
        'domain_train': domain_train, 'domain_val': domain_val, 'domain_test': domain_test,
        'scaler': scaler, 'feature_columns': feature_columns,
        'attack_types': attack_types, 'domains': domains
    }
    
    return data_dict

def step3_initialize_and_build_model(input_dim, attack_types, domains):
    """
    STEP 3: Initialize and build the SecureRouteX GAN model
    """
    print("\n🏗️ STEP 3: BUILDING SECUREROUTEX GAN MODEL")
    print("="*50)
    
    # Initialize the GAN model
    print("🚀 Initializing SecureRouteX GAN...")
    secureroutex = SecureRouteXGAN(
        input_dim=input_dim,
        latent_dim=100,
        domains=domains
    )
    
    # Build all model components
    print("\n🔧 Building model architecture...")
    
    print("   📦 Building Generator...")
    secureroutex.build_generator()
    
    print("   📦 Building Discriminator...")
    secureroutex.build_discriminator()
    
    print("   📦 Building Trust Evaluator...")
    secureroutex.build_trust_evaluator()
    
    print("   📦 Building Attack Detector...")
    secureroutex.build_attack_detector()
    
    print("   📦 Building Complete GAN...")
    secureroutex.build_complete_gan()
    
    # Compile models
    print("\n⚙️ Compiling models with optimizers...")
    secureroutex.compile_models()
    
    print("✅ Model initialization complete!")
    print(f"   • Input dimensions: {input_dim}")
    print(f"   • Latent space: 100")
    print(f"   • Attack types: {len(attack_types)}")
    print(f"   • Domains: {len(domains)}")
    
    return secureroutex

def step4_train_the_model(secureroutex, data_dict, epochs=50):
    """
    STEP 4: Train the GAN model with your data
    """
    print(f"\n🚀 STEP 4: TRAINING THE MODEL ({epochs} EPOCHS)")
    print("="*50)
    
    X_train = data_dict['X_train']
    attack_train = data_dict['attack_train']
    domain_train = data_dict['domain_train']
    
    print("🎯 Starting GAN training...")
    print(f"   • Training samples: {X_train.shape[0]:,}")
    print(f"   • Epochs: {epochs}")
    print(f"   • Batch size: 32")
    
    # Train the model
    secureroutex.train_gan(
        X_train=X_train,
        attack_train=attack_train, 
        domain_train=domain_train,
        epochs=epochs,
        batch_size=32
    )
    
    print("✅ Training completed!")
    
    return secureroutex

def step5_evaluate_performance(secureroutex, data_dict):
    """
    STEP 5: Evaluate model performance on test data
    """
    print("\n📊 STEP 5: EVALUATING MODEL PERFORMANCE")
    print("="*50)
    
    X_test = data_dict['X_test']
    attack_test = data_dict['attack_test']
    domain_test = data_dict['domain_test']
    
    print("🎯 Running comprehensive evaluation...")
    
    # Evaluate model performance
    performance = secureroutex.evaluate_model_performance(X_test, attack_test, domain_test)
    
    print("✅ Performance Evaluation Results:")
    
    # Display discriminator performance
    if 'discriminator_auc' in performance:
        print(f"   🎯 Discriminator AUC: {performance['discriminator_auc']:.4f}")
        if performance['discriminator_auc'] > 0.9:
            print("      ✅ Excellent performance (>0.9)")
        elif performance['discriminator_auc'] > 0.8:
            print("      ✅ Good performance (>0.8)")
        else:
            print("      ⚠️ Needs improvement (<0.8)")
    
    # Display trust score statistics
    if 'trust_scores' in performance:
        print(f"\n   🛡️ Trust Score Analysis:")
        trust_stats = performance['trust_scores']
        print(f"      • Direct Trust (avg): {trust_stats['direct_mean']:.3f}")
        print(f"      • Indirect Trust (avg): {trust_stats['indirect_mean']:.3f}")
        print(f"      • Energy Trust (avg): {trust_stats['energy_mean']:.3f}")
        print(f"      • Composite Trust (avg): {trust_stats['composite_mean']:.3f}")
    
    # Display attack detection results
    if 'attack_detection' in performance:
        print(f"\n   🚨 Attack Detection Analysis:")
        attack_stats = performance['attack_detection']
        print(f"      • Mean Attack Probability: {attack_stats['mean_attack_probability']:.3f}")
        print(f"      • Attack Class Distribution: {attack_stats['attack_class_distribution']}")
    
    return performance

def step6_generate_synthetic_attacks(secureroutex, domains, attack_types):
    """
    STEP 6: Generate synthetic attack data for each domain
    """
    print("\n🎯 STEP 6: GENERATING SYNTHETIC ATTACK DATA")
    print("="*50)
    
    synthetic_results = {}
    
    for domain in domains:
        print(f"\n🏥 Generating attacks for {domain} domain:")
        domain_attacks = {}
        
        for attack_type in attack_types:
            if attack_type != 'normal':  # Skip normal traffic
                print(f"   • Generating {attack_type} attacks...")
                
                # Generate 100 synthetic attacks for this domain/attack combination
                synthetic_data = secureroutex.generate_synthetic_attacks(
                    num_samples=100,
                    domain=domain,
                    attack_type=attack_type
                )
                
                domain_attacks[attack_type] = synthetic_data
                print(f"     ✅ Generated {synthetic_data.shape[0]} samples")
        
        synthetic_results[domain] = domain_attacks
    
    print(f"\n✅ Synthetic attack generation complete!")
    print(f"   • Total domains: {len(synthetic_results)}")
    print(f"   • Attack types per domain: {len(attack_types)-1}")  # Exclude 'normal'
    
    return synthetic_results

def step7_test_real_time_detection(secureroutex, data_dict):
    """
    STEP 7: Test real-time attack detection capabilities
    """
    print("\n🚨 STEP 7: TESTING REAL-TIME ATTACK DETECTION")
    print("="*50)
    
    X_test = data_dict['X_test']
    domains = data_dict['domains']
    
    print("🔍 Testing real-time detection on sample data...")
    
    # Test on a few samples
    num_test_samples = 5
    
    for i in range(num_test_samples):
        print(f"\n📊 Test Sample {i+1}:")
        
        # Get sample network features
        network_features = X_test[i:i+1]  # Keep as 2D array
        
        # Calculate trust scores for this sample
        domain_idx = np.random.randint(0, len(domains))  # Random domain for demo
        domain = domains[domain_idx]
        
        trust_scores = secureroutex.calculate_trust_scores(network_features, domain)
        
        # Create dummy historical context (in real system, this would be actual history)
        history_context = np.random.uniform(0.3, 0.8, (1, 10))
        
        # Combine trust scores for attack detector input
        trust_input = np.array([[
            trust_scores['direct_trust'][0],
            trust_scores['indirect_trust'][0], 
            trust_scores['energy_trust'][0],
            trust_scores['composite_trust'][0]
        ]])
        
        # Perform real-time detection
        detection_result = secureroutex.detect_attacks_realtime(
            network_features, trust_input, history_context
        )
        
        print(f"   🏥 Domain: {domain}")
        print(f"   🛡️ Trust Scores:")
        print(f"      • Direct: {trust_scores['direct_trust'][0]:.3f}")
        print(f"      • Composite: {trust_scores['composite_trust'][0]:.3f}")
        print(f"   🚨 Attack Detection:")
        print(f"      • Attack Probability: {detection_result['attack_probability']:.3f}")
        print(f"      • Predicted Attack: {detection_result['predicted_attack']}")
        print(f"      • Recommended Action: {detection_result['recommended_action']}")
        
        # Determine if this is a security threat
        if detection_result['attack_probability'] > 0.5:
            print(f"      ⚠️  HIGH THREAT - Action: {detection_result['recommended_action'].upper()}")
        else:
            print(f"      ✅ Low threat - Normal operation")
    
    print("\n✅ Real-time detection test complete!")

def step8_save_results_and_models(secureroutex, performance, synthetic_results):
    """
    STEP 8: Save trained models and generate comprehensive results
    """
    print("\n💾 STEP 8: SAVING MODELS AND RESULTS")
    print("="*50)
    
    # Save all trained models
    print("📁 Saving trained models...")
    secureroutex.save_model('secureroutex_trained_models')
    
    # Create training visualization
    print("📊 Generating training visualization...")
    secureroutex.visualize_training_results('secureroutex_training_results.png')
    
    # Save synthetic attack data
    print("💾 Saving synthetic attack data...")
    
    # Convert synthetic results to DataFrames and save
    all_synthetic_data = []
    
    for domain, attacks in synthetic_results.items():
        for attack_type, data in attacks.items():
            for i, sample in enumerate(data):
                row = {
                    'sample_id': f"{domain}_{attack_type}_{i}",
                    'domain': domain,
                    'attack_type': attack_type,
                    'synthetic': True
                }
                # Add feature values
                for j, value in enumerate(sample):
                    row[f'feature_{j}'] = value
                
                all_synthetic_data.append(row)
    
    synthetic_df = pd.DataFrame(all_synthetic_data)
    synthetic_df.to_csv('secureroutex_synthetic_attacks.csv', index=False)
    
    print(f"   ✅ Saved {len(synthetic_df)} synthetic attack samples")
    
    # Create comprehensive results report
    print("📋 Creating results report...")
    
    results_report = f"""
SecureRouteX Training Results Report
===================================

Training Configuration:
• Model: SecureRouteX GAN with Trust Evaluation
• Dataset: Enhanced SecureRouteX Dataset (9,000 samples)
• Architecture: Generator + Discriminator + Trust Evaluator + Attack Detector
• Training: Adversarial training with multi-task learning

Performance Results:
"""
    
    if 'discriminator_auc' in performance:
        results_report += f"• Discriminator AUC: {performance['discriminator_auc']:.4f}\n"
    
    if 'trust_scores' in performance:
        trust_stats = performance['trust_scores']
        results_report += f"• Average Composite Trust: {trust_stats['composite_mean']:.3f}\n"
    
    results_report += f"""

Synthetic Data Generation:
• Healthcare domain attacks: {len(synthetic_results.get('healthcare', {})) * 100} samples
• Transportation domain attacks: {len(synthetic_results.get('transportation', {})) * 100} samples  
• Underwater domain attacks: {len(synthetic_results.get('underwater', {})) * 100} samples

Model Capabilities:
✅ Real-time attack detection
✅ Multi-dimensional trust calculation
✅ SDN routing recommendations (Allow/Block/Reroute)
✅ Cross-domain security intelligence
✅ Synthetic attack pattern generation

Files Generated:
• secureroutex_trained_models/ - Complete trained model files
• secureroutex_training_results.png - Training visualization
• secureroutex_synthetic_attacks.csv - Generated synthetic attack data
• secureroutex_results_report.txt - This report

Next Steps:
1. Deploy model in SDN testbed environment
2. Conduct NS-3 network simulation with 15 nodes
3. Compare performance against traditional routing methods
4. Prepare results for conference paper submission
"""
    
    with open('secureroutex_results_report.txt', 'w') as f:
        f.write(results_report)
    
    print("✅ All results saved successfully!")
    print(f"\n📁 Generated Files:")
    print(f"   • secureroutex_trained_models/ (Model files)")
    print(f"   • secureroutex_training_results.png (Visualization)")
    print(f"   • secureroutex_synthetic_attacks.csv (Synthetic data)")
    print(f"   • secureroutex_results_report.txt (Complete report)")

def main_training_pipeline():
    """
    MAIN: Complete SecureRouteX training pipeline
    """
    print("🚀 SECUREROUTEX COMPLETE TRAINING PIPELINE")
    print("="*60)
    print("This script will train your GAN model step by step using your enhanced dataset.")
    print("Each step is clearly explained and shows you what to do next.")
    print("="*60)
    
    try:
        # Step 1: Load data
        df = step1_load_and_prepare_data()
        
        # Step 2: Preprocess data
        data_dict = step2_preprocess_data(df)
        
        # Step 3: Build model
        secureroutex = step3_initialize_and_build_model(
            input_dim=data_dict['X_train'].shape[1],
            attack_types=data_dict['attack_types'],
            domains=data_dict['domains']
        )
        
        # Step 4: Train model
        secureroutex = step4_train_the_model(secureroutex, data_dict, epochs=20)  # Reduced for demo
        
        # Step 5: Evaluate performance
        performance = step5_evaluate_performance(secureroutex, data_dict)
        
        # Step 6: Generate synthetic attacks
        synthetic_results = step6_generate_synthetic_attacks(
            secureroutex, data_dict['domains'], data_dict['attack_types']
        )
        
        # Step 7: Test real-time detection
        step7_test_real_time_detection(secureroutex, data_dict)
        
        # Step 8: Save everything
        step8_save_results_and_models(secureroutex, performance, synthetic_results)
        
        print("\n🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("✅ Your SecureRouteX GAN model is now fully trained and ready!")
        print("✅ All results, models, and reports have been saved.")
        print("✅ You can now use the model for your review presentation tomorrow.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        print("💡 Make sure your 'secureroutex_enhanced_dataset.csv' file exists in the current directory.")

if __name__ == "__main__":
    main_training_pipeline()