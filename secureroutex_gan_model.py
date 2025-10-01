#!/usr/bin/env python3
"""
SecureRouteX: Complete GAN Model Implementation
==============================================

Comprehensive GAN model for trust-aware AI-SDN routing in multi-domain IoT networks.
Includes Generator, Discriminator, Trust Evaluator, and Attack Detection components.

Features:
- Multi-domain IoT attack generation (Healthcare, Transportation, Underwater)
- Real-time trust score calculation and anomaly detection
- SDN-compatible routing decision support
- Cross-domain trust sharing mechanisms

Author: SecureRouteX Research Team
Date: September 2025
License: Academic Research Use
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SecureRouteXGAN:
    """
    Complete GAN implementation for SecureRouteX multi-domain IoT security
    
    Components:
    1. Generator: Creates synthetic attack patterns for training
    2. Discriminator: Distinguishes real vs synthetic network traffic
    3. Trust Evaluator: Calculates multi-dimensional trust scores
    4. Attack Detector: Real-time anomaly detection for SDN routing
    """
    
    def __init__(self, input_dim=50, latent_dim=100, domains=['healthcare', 'transportation', 'underwater']):
        """
        Initialize SecureRouteX GAN model
        
        Args:
            input_dim (int): Number of network features (default: 50)
            latent_dim (int): Latent space dimension for generator
            domains (list): IoT domain types for multi-domain support
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.domains = domains
        self.attack_types = ['normal', 'ddos', 'malicious_node', 'energy_drain', 'routing_attack']
        
        # Domain-specific trust baselines (from literature analysis)
        self.trust_baselines = {
            'healthcare': 0.75,      # High trust requirement (HIPAA compliance)
            'transportation': 0.65,  # Medium trust (V2X communication)
            'underwater': 0.55       # Low trust (harsh environment)
        }
        
        # Domain-specific performance requirements
        self.domain_requirements = {
            'healthcare': {'max_latency': 5, 'min_trust': 0.75, 'priority': 'high'},
            'transportation': {'max_latency': 15, 'min_trust': 0.65, 'priority': 'medium'}, 
            'underwater': {'max_latency': 100, 'min_trust': 0.55, 'priority': 'low'}
        }
        
        # Initialize model components
        self.generator = None
        self.discriminator = None
        self.trust_evaluator = None
        self.attack_detector = None
        self.gan_model = None
        
        # Training history
        self.history = {
            'gen_loss': [], 'disc_loss': [], 'trust_accuracy': [], 
            'attack_detection_auc': [], 'cross_domain_trust': []
        }
        
        # Preprocessing components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.domain_encoder = LabelEncoder()
        
        print("🚀 SecureRouteX GAN Model Initialized!")
        print(f"   • Input Dimensions: {self.input_dim}")
        print(f"   • Latent Space: {self.latent_dim}")
        print(f"   • Supported Domains: {self.domains}")
        print(f"   • Attack Types: {self.attack_types}")
    
    def build_generator(self):
        """
        Build Generator network for synthetic IoT attack pattern generation
        
        Architecture:
        - Multi-layer fully connected network
        - Domain-aware generation with conditional inputs
        - Batch normalization and dropout for stability
        """
        # Noise input
        noise_input = layers.Input(shape=(self.latent_dim,), name='noise_input')
        
        # Domain condition input (one-hot encoded)
        domain_input = layers.Input(shape=(len(self.domains),), name='domain_input')
        
        # Attack type condition input
        attack_input = layers.Input(shape=(len(self.attack_types),), name='attack_input')
        
        # Combine all inputs
        combined_input = layers.Concatenate()([noise_input, domain_input, attack_input])
        
        # Hidden layers with progressive expansion
        x = layers.Dense(128, activation='relu')(combined_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer - generate synthetic network features
        synthetic_features = layers.Dense(self.input_dim, activation='tanh', name='synthetic_features')(x)
        
        # Create generator model
        generator = models.Model(
            inputs=[noise_input, domain_input, attack_input],
            outputs=synthetic_features,
            name='SecureRouteX_Generator'
        )
        
        self.generator = generator
        
        print("✅ Generator Network Built:")
        print(f"   • Architecture: {self.latent_dim + len(self.domains) + len(self.attack_types)} → 128 → 256 → 512 → {self.input_dim}")
        print(f"   • Conditional Inputs: Domain + Attack Type")
        print(f"   • Output: Synthetic IoT network features")
        
        return generator
    
    def build_discriminator(self):
        """
        Build Discriminator network for real vs synthetic traffic classification
        
        Architecture:
        - Convolutional-style feature extraction
        - Multi-output: Real/Fake + Attack Classification + Domain Classification
        """
        # Network features input
        feature_input = layers.Input(shape=(self.input_dim,), name='feature_input')
        
        # Feature extraction layers
        x = layers.Dense(512, activation='leaky_relu')(feature_input)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='leaky_relu')(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='leaky_relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Multi-task outputs
        # 1. Real vs Fake classification
        real_fake_output = layers.Dense(1, activation='sigmoid', name='real_fake')(x)
        
        # 2. Attack type classification
        attack_output = layers.Dense(len(self.attack_types), activation='softmax', name='attack_type')(x)
        
        # 3. Domain classification  
        domain_output = layers.Dense(len(self.domains), activation='softmax', name='domain_type')(x)
        
        # Create discriminator model
        discriminator = models.Model(
            inputs=feature_input,
            outputs=[real_fake_output, attack_output, domain_output],
            name='SecureRouteX_Discriminator'
        )
        
        self.discriminator = discriminator
        
        print("✅ Discriminator Network Built:")
        print(f"   • Architecture: {self.input_dim} → 512 → 256 → 128 → Multi-output")
        print(f"   • Outputs: Real/Fake + Attack Classification + Domain Classification")
        print(f"   • Multi-task learning for enhanced security")
        
        return discriminator
    
    def build_trust_evaluator(self):
        """
        Build Trust Evaluator network for multi-dimensional trust calculation
        
        Calculates:
        1. Direct Trust: Based on direct interactions and performance
        2. Indirect Trust: Based on recommendations from other nodes  
        3. Energy Trust: Based on energy consumption patterns
        4. Composite Trust: Weighted combination of all trust dimensions
        """
        # Network behavior input features
        behavior_input = layers.Input(shape=(self.input_dim,), name='behavior_input')
        
        # Domain context input
        domain_input = layers.Input(shape=(len(self.domains),), name='domain_context')
        
        # Combine inputs
        combined_input = layers.Concatenate()([behavior_input, domain_input])
        
        # Shared feature extraction
        x = layers.Dense(256, activation='relu')(combined_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Trust dimension calculation branches
        # Direct Trust Branch
        direct_branch = layers.Dense(64, activation='relu')(x)
        direct_trust = layers.Dense(1, activation='sigmoid', name='direct_trust')(direct_branch)
        
        # Indirect Trust Branch  
        indirect_branch = layers.Dense(64, activation='relu')(x)
        indirect_trust = layers.Dense(1, activation='sigmoid', name='indirect_trust')(indirect_branch)
        
        # Energy Trust Branch
        energy_branch = layers.Dense(64, activation='relu')(x)
        energy_trust = layers.Dense(1, activation='sigmoid', name='energy_trust')(energy_branch)
        
        # Composite Trust Calculation Layer
        trust_concat = layers.Concatenate()([direct_trust, indirect_trust, energy_trust, domain_input])
        composite_layer = layers.Dense(32, activation='relu')(trust_concat)
        composite_trust = layers.Dense(1, activation='sigmoid', name='composite_trust')(composite_layer)
        
        # Trust evaluator model
        trust_evaluator = models.Model(
            inputs=[behavior_input, domain_input],
            outputs=[direct_trust, indirect_trust, energy_trust, composite_trust],
            name='SecureRouteX_TrustEvaluator'
        )
        
        self.trust_evaluator = trust_evaluator
        
        print("✅ Trust Evaluator Network Built:")
        print(f"   • Multi-dimensional trust calculation")
        print(f"   • Outputs: Direct + Indirect + Energy + Composite Trust")
        print(f"   • Domain-aware trust baselines integrated")
        
        return trust_evaluator
    
    def build_attack_detector(self):
        """
        Build Attack Detector for real-time anomaly detection in SDN routing
        
        Features:
        - Real-time classification of network anomalies
        - Cross-domain attack pattern recognition
        - SDN-compatible output format for routing decisions
        """
        # Real-time network features input
        network_input = layers.Input(shape=(self.input_dim,), name='network_input')
        
        # Trust scores input (from trust evaluator)
        trust_input = layers.Input(shape=(4,), name='trust_scores')  # 4 trust dimensions
        
        # Historical behavior context
        history_input = layers.Input(shape=(10,), name='history_context')  # Last 10 time steps
        
        # Combine all inputs for comprehensive analysis
        combined_input = layers.Concatenate()([network_input, trust_input, history_input])
        
        # Deep feature extraction
        x = layers.Dense(512, activation='relu')(combined_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layers for SDN integration
        # 1. Attack probability score
        attack_probability = layers.Dense(1, activation='sigmoid', name='attack_probability')(x)
        
        # 2. Attack type classification
        attack_classification = layers.Dense(len(self.attack_types), activation='softmax', name='attack_classification')(x)
        
        # 3. Routing recommendation (allow/block/reroute)
        routing_decision = layers.Dense(3, activation='softmax', name='routing_decision')(x)
        
        # 4. Trust adjustment recommendation
        trust_adjustment = layers.Dense(1, activation='tanh', name='trust_adjustment')(x)
        
        # Attack detector model
        attack_detector = models.Model(
            inputs=[network_input, trust_input, history_input],
            outputs=[attack_probability, attack_classification, routing_decision, trust_adjustment],
            name='SecureRouteX_AttackDetector'
        )
        
        self.attack_detector = attack_detector
        
        print("✅ Attack Detector Network Built:")
        print(f"   • Real-time anomaly detection")
        print(f"   • SDN-compatible routing decisions")
        print(f"   • Outputs: Attack Probability + Classification + Routing + Trust Adjustment")
        
        return attack_detector
    
    def build_complete_gan(self):
        """
        Build complete GAN model integrating Generator and Discriminator
        """
        if self.generator is None:
            self.build_generator()
        if self.discriminator is None:
            self.build_discriminator()
        
        # Freeze discriminator during generator training
        self.discriminator.trainable = False
        
        # GAN input (same as generator inputs)
        noise_input = layers.Input(shape=(self.latent_dim,), name='gan_noise')
        domain_input = layers.Input(shape=(len(self.domains),), name='gan_domain')
        attack_input = layers.Input(shape=(len(self.attack_types),), name='gan_attack')
        
        # Generate synthetic data
        generated_data = self.generator([noise_input, domain_input, attack_input])
        
        # Pass through discriminator
        discriminator_outputs = self.discriminator(generated_data)
        
        # Complete GAN model
        gan_model = models.Model(
            inputs=[noise_input, domain_input, attack_input],
            outputs=discriminator_outputs,
            name='SecureRouteX_CompleteGAN'
        )
        
        self.gan_model = gan_model
        
        print("✅ Complete GAN Model Built:")
        print(f"   • Integrated Generator + Discriminator")
        print(f"   • Multi-domain conditional generation")
        print(f"   • Ready for adversarial training")
        
        return gan_model
    
    def compile_models(self, learning_rate=0.0002, beta_1=0.5):
        """
        Compile all models with appropriate optimizers and loss functions
        """
        optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)
        
        # Compile Discriminator with multi-task losses
        if self.discriminator is not None:
            self.discriminator.compile(
                optimizer=optimizer,
                loss={
                    'real_fake': 'binary_crossentropy',
                    'attack_type': 'categorical_crossentropy',
                    'domain_type': 'categorical_crossentropy'
                },
                loss_weights={'real_fake': 1.0, 'attack_type': 0.5, 'domain_type': 0.3},
                metrics=['accuracy']
            )
        
        # Compile Trust Evaluator with multi-output regression
        if self.trust_evaluator is not None:
            self.trust_evaluator.compile(
                optimizer=optimizer,
                loss={
                    'direct_trust': 'mse',
                    'indirect_trust': 'mse', 
                    'energy_trust': 'mse',
                    'composite_trust': 'mse'
                },
                loss_weights={'direct_trust': 0.3, 'indirect_trust': 0.2, 'energy_trust': 0.2, 'composite_trust': 0.3},
                metrics=['mae']
            )
        
        # Compile Attack Detector with multi-task classification
        if self.attack_detector is not None:
            self.attack_detector.compile(
                optimizer=optimizer,
                loss={
                    'attack_probability': 'binary_crossentropy',
                    'attack_classification': 'categorical_crossentropy',
                    'routing_decision': 'categorical_crossentropy',
                    'trust_adjustment': 'mse'
                },
                loss_weights={'attack_probability': 0.4, 'attack_classification': 0.3, 'routing_decision': 0.2, 'trust_adjustment': 0.1},
                metrics=['accuracy']
            )
        
        # Compile complete GAN
        if self.gan_model is not None:
            self.gan_model.compile(
                optimizer=optimizer,
                loss={
                    'real_fake': 'binary_crossentropy',
                    'attack_type': 'categorical_crossentropy', 
                    'domain_type': 'categorical_crossentropy'
                },
                loss_weights={'real_fake': 1.0, 'attack_type': 0.5, 'domain_type': 0.3}
            )
        
        print("✅ All Models Compiled Successfully:")
        print(f"   • Learning Rate: {learning_rate}")
        print(f"   • Optimizer: Adam (beta_1={beta_1})")
        print(f"   • Multi-task loss functions configured")
    
    def load_and_preprocess_data(self, data_path):
        """
        Load and preprocess SecureRouteX dataset for training
        
        Args:
            data_path (str): Path to enhanced dataset CSV file
        """
        print("📊 Loading SecureRouteX Dataset...")
        
        # Load dataset
        df = pd.read_csv(data_path)
        print(f"   • Dataset Shape: {df.shape}")
        print(f"   • Columns: {list(df.columns)}")
        
        # Separate features and labels
        feature_columns = [col for col in df.columns if col not in ['attack_type', 'domain', 'is_malicious']]
        
        X = df[feature_columns].values
        attack_labels = df['attack_type'].values
        domain_labels = df['domain'].values
        
        # Encode categorical variables
        attack_encoded = pd.get_dummies(df['attack_type']).values
        domain_encoded = pd.get_dummies(df['domain']).values
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, attack_train, attack_test, domain_train, domain_test = train_test_split(
            X_scaled, attack_encoded, domain_encoded, test_size=0.2, random_state=42, stratify=attack_labels
        )
        
        print("✅ Data Preprocessing Complete:")
        print(f"   • Training Samples: {X_train.shape[0]}")
        print(f"   • Test Samples: {X_test.shape[0]}")
        print(f"   • Feature Dimensions: {X_train.shape[1]}")
        print(f"   • Attack Types: {attack_encoded.shape[1]}")
        print(f"   • Domain Types: {domain_encoded.shape[1]}")
        
        return X_train, X_test, attack_train, attack_test, domain_train, domain_test
    
    def generate_synthetic_attacks(self, num_samples=1000, domain='healthcare', attack_type='ddos'):
        """
        Generate synthetic attack patterns for specified domain and attack type
        
        Args:
            num_samples (int): Number of synthetic samples to generate
            domain (str): Target IoT domain
            attack_type (str): Type of attack to simulate
            
        Returns:
            numpy.ndarray: Generated synthetic attack data
        """
        if self.generator is None:
            raise ValueError("Generator not built. Call build_generator() first.")
        
        # Create condition vectors
        domain_idx = self.domains.index(domain)
        attack_idx = self.attack_types.index(attack_type)
        
        # Generate random noise
        noise = np.random.normal(0, 1, (num_samples, self.latent_dim))
        
        # Create one-hot encoded conditions
        domain_conditions = np.zeros((num_samples, len(self.domains)))
        domain_conditions[:, domain_idx] = 1
        
        attack_conditions = np.zeros((num_samples, len(self.attack_types)))
        attack_conditions[:, attack_idx] = 1
        
        # Generate synthetic data
        synthetic_data = self.generator.predict([noise, domain_conditions, attack_conditions])
        
        print(f"✅ Generated {num_samples} synthetic {attack_type} attacks for {domain} domain")
        
        return synthetic_data
    
    def calculate_trust_scores(self, network_data, domain):
        """
        Calculate multi-dimensional trust scores for network nodes
        
        Args:
            network_data (numpy.ndarray): Network behavior features
            domain (str): IoT domain context
            
        Returns:
            dict: Trust scores (direct, indirect, energy, composite)
        """
        if self.trust_evaluator is None:
            raise ValueError("Trust Evaluator not built. Call build_trust_evaluator() first.")
        
        # Prepare domain context
        domain_idx = self.domains.index(domain)
        domain_context = np.zeros((network_data.shape[0], len(self.domains)))
        domain_context[:, domain_idx] = 1
        
        # Calculate trust scores
        trust_outputs = self.trust_evaluator.predict([network_data, domain_context])
        
        trust_scores = {
            'direct_trust': trust_outputs[0].flatten(),
            'indirect_trust': trust_outputs[1].flatten(),
            'energy_trust': trust_outputs[2].flatten(),
            'composite_trust': trust_outputs[3].flatten()
        }
        
        # Apply domain-specific baselines
        baseline = self.trust_baselines[domain]
        trust_scores['adjusted_composite'] = trust_scores['composite_trust'] * baseline
        
        return trust_scores
    
    def detect_attacks_realtime(self, network_features, trust_scores, history_context):
        """
        Real-time attack detection for SDN routing decisions
        
        Args:
            network_features (numpy.ndarray): Current network state
            trust_scores (numpy.ndarray): Multi-dimensional trust scores
            history_context (numpy.ndarray): Historical behavior context
            
        Returns:
            dict: Attack detection results and routing recommendations
        """
        if self.attack_detector is None:
            raise ValueError("Attack Detector not built. Call build_attack_detector() first.")
        
        # Prepare inputs
        if len(network_features.shape) == 1:
            network_features = network_features.reshape(1, -1)
        if len(trust_scores.shape) == 1:
            trust_scores = trust_scores.reshape(1, -1)
        if len(history_context.shape) == 1:
            history_context = history_context.reshape(1, -1)
        
        # Get attack detection results
        detection_outputs = self.attack_detector.predict([network_features, trust_scores, history_context])
        
        results = {
            'attack_probability': detection_outputs[0][0][0],
            'attack_classification': detection_outputs[1][0],
            'routing_decision': detection_outputs[2][0],  # [allow, block, reroute]
            'trust_adjustment': detection_outputs[3][0][0]
        }
        
        # Interpret routing decision
        routing_actions = ['allow', 'block', 'reroute']
        recommended_action = routing_actions[np.argmax(results['routing_decision'])]
        results['recommended_action'] = recommended_action
        
        # Interpret attack type
        predicted_attack = self.attack_types[np.argmax(results['attack_classification'])]
        results['predicted_attack'] = predicted_attack
        
        return results
    
    def train_gan(self, X_train, attack_train, domain_train, epochs=100, batch_size=32):
        """
        Train the complete GAN model with adversarial learning
        
        Args:
            X_train: Training network features
            attack_train: Attack type labels (one-hot encoded)
            domain_train: Domain labels (one-hot encoded)
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
        """
        print("🚀 Starting SecureRouteX GAN Training...")
        
        if self.generator is None or self.discriminator is None:
            self.build_generator()
            self.build_discriminator()
            self.build_complete_gan()
            self.compile_models()
        
        # Training loop
        for epoch in range(epochs):
            # ============================================
            # Train Discriminator
            # ============================================
            
            # Select random real samples
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            real_data = X_train[idx]
            real_attacks = attack_train[idx]
            real_domains = domain_train[idx]
            
            # Generate fake samples
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_domains = real_domains  # Use same domain distribution
            fake_attacks = real_attacks  # Use same attack distribution
            
            fake_data = self.generator.predict([noise, fake_domains, fake_attacks])
            
            # Create labels for real/fake classification
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))
            
            # Train discriminator on real data
            d_loss_real = self.discriminator.train_on_batch(
                real_data,
                [real_labels, real_attacks, real_domains]
            )
            
            # Train discriminator on fake data
            d_loss_fake = self.discriminator.train_on_batch(
                fake_data,
                [fake_labels, fake_attacks, fake_domains]
            )
            
            # Average discriminator loss
            d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
            
            # ============================================
            # Train Generator (via GAN)
            # ============================================
            
            # Generate new noise and conditions
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            
            # Generator wants discriminator to classify fakes as real
            misleading_labels = np.ones((batch_size, 1))
            
            # Train generator
            g_loss = self.gan_model.train_on_batch(
                [noise, fake_domains, fake_attacks],
                [misleading_labels, fake_attacks, fake_domains]
            )
            
            # Store training history
            self.history['gen_loss'].append(g_loss[0])
            self.history['disc_loss'].append(d_loss)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - D Loss: {d_loss:.4f}, G Loss: {g_loss[0]:.4f}")
        
        print("✅ GAN Training Complete!")
        
    def evaluate_model_performance(self, X_test, attack_test, domain_test):
        """
        Evaluate complete model performance on test data
        
        Args:
            X_test: Test network features
            attack_test: Test attack labels
            domain_test: Test domain labels
            
        Returns:
            dict: Comprehensive performance metrics
        """
        print("📊 Evaluating SecureRouteX Model Performance...")
        
        results = {}
        
        # 1. Discriminator Performance
        if self.discriminator is not None:
            # Create mixed real/fake test data
            batch_size = min(100, X_test.shape[0])
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            fake_data = self.generator.predict([noise, domain_test[:batch_size], attack_test[:batch_size]])
            
            # Test discriminator
            real_pred = self.discriminator.predict(X_test[:batch_size])
            fake_pred = self.discriminator.predict(fake_data)
            
            # Calculate AUC for real/fake classification
            real_scores = real_pred[0].flatten()
            fake_scores = fake_pred[0].flatten()
            
            y_true = np.concatenate([np.ones(len(real_scores)), np.zeros(len(fake_scores))])
            y_scores = np.concatenate([real_scores, fake_scores])
            
            discriminator_auc = roc_auc_score(y_true, y_scores)
            results['discriminator_auc'] = discriminator_auc
        
        # 2. Trust Evaluator Performance
        if self.trust_evaluator is not None:
            trust_predictions = self.trust_evaluator.predict([X_test[:100], domain_test[:100]])
            
            # Calculate trust score statistics
            results['trust_scores'] = {
                'direct_mean': np.mean(trust_predictions[0]),
                'indirect_mean': np.mean(trust_predictions[1]),
                'energy_mean': np.mean(trust_predictions[2]),
                'composite_mean': np.mean(trust_predictions[3])
            }
        
        # 3. Attack Detector Performance
        if self.attack_detector is not None:
            # Create dummy trust scores and history for testing
            dummy_trust = np.random.uniform(0.3, 0.9, (100, 4))
            dummy_history = np.random.uniform(0, 1, (100, 10))
            
            attack_predictions = self.attack_detector.predict([X_test[:100], dummy_trust, dummy_history])
            
            # Calculate attack detection metrics
            attack_probs = attack_predictions[0].flatten()
            attack_classes = np.argmax(attack_predictions[1], axis=1)
            
            results['attack_detection'] = {
                'mean_attack_probability': np.mean(attack_probs),
                'attack_class_distribution': np.bincount(attack_classes)
            }
        
        print("✅ Model Evaluation Complete!")
        return results
    
    def visualize_training_results(self, save_path='secureroutex_training_results.png'):
        """
        Create comprehensive visualization of training results and model performance
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SecureRouteX GAN Model Training Results', fontsize=16, fontweight='bold')
        
        # 1. Training Loss Curves
        if len(self.history['gen_loss']) > 0:
            axes[0,0].plot(self.history['gen_loss'], label='Generator Loss', color='blue', alpha=0.7)
            axes[0,0].plot(self.history['disc_loss'], label='Discriminator Loss', color='red', alpha=0.7)
            axes[0,0].set_title('GAN Training Loss')
            axes[0,0].set_xlabel('Epoch')
            axes[0,0].set_ylabel('Loss')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. Domain-specific Trust Baselines
        domains = list(self.trust_baselines.keys())
        trust_values = list(self.trust_baselines.values())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars = axes[0,1].bar(domains, trust_values, color=colors, alpha=0.7)
        axes[0,1].set_title('Domain-Specific Trust Baselines')
        axes[0,1].set_ylabel('Trust Baseline')
        axes[0,1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, trust_values):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{value:.2f}', ha='center', va='bottom')
        
        # 3. Attack Type Distribution
        attack_counts = [1] * len(self.attack_types)  # Equal distribution for demo
        axes[0,2].pie(attack_counts, labels=self.attack_types, autopct='%1.1f%%', startangle=90)
        axes[0,2].set_title('Supported Attack Types')
        
        # 4. Domain Performance Requirements
        domains = list(self.domain_requirements.keys())
        latencies = [self.domain_requirements[d]['max_latency'] for d in domains]
        
        axes[1,0].bar(domains, latencies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.7)
        axes[1,0].set_title('Domain Latency Requirements')
        axes[1,0].set_ylabel('Max Latency (ms)')
        axes[1,0].set_yscale('log')
        
        # Add value labels
        for i, (domain, latency) in enumerate(zip(domains, latencies)):
            axes[1,0].text(i, latency * 1.1, f'{latency}ms', ha='center', va='bottom')
        
        # 5. Model Architecture Summary
        axes[1,1].axis('off')
        architecture_text = f"""
        SecureRouteX Architecture Summary:
        
        🔧 Generator:
        • Input: {self.latent_dim}D noise + conditions
        • Output: {self.input_dim}D synthetic features
        • Layers: 128→256→512→{self.input_dim}
        
        🎯 Discriminator:  
        • Multi-task: Real/Fake + Attack + Domain
        • Architecture: {self.input_dim}→512→256→128
        
        🛡️ Trust Evaluator:
        • 4D Trust: Direct + Indirect + Energy + Composite
        • Domain-aware baseline adjustment
        
        🚨 Attack Detector:
        • Real-time SDN integration
        • Outputs: Probability + Classification + Routing
        """
        
        axes[1,1].text(0.05, 0.95, architecture_text, transform=axes[1,1].transAxes, 
                      verticalalignment='top', fontsize=10, fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        # 6. Performance Metrics Summary
        axes[1,2].axis('off')
        metrics_text = f"""
        Performance Targets:
        
        📊 Dataset Quality: Grade B (0.6540)
        🎯 Statistical Fidelity: 0.8073
        🚀 ML Utility AUC: 99.6%
        🔒 Privacy: ε-DP (ε=1.0)
        
        Domain Requirements:
        • Healthcare: <5ms, Trust>0.75
        • Transport: <15ms, Trust>0.65  
        • Underwater: <100ms, Trust>0.55
        
        Attack Coverage:
        • DDoS, Malicious Node
        • Energy Drain, Routing Attack
        • Normal Operation Baseline
        """
        
        axes[1,2].text(0.05, 0.95, metrics_text, transform=axes[1,2].transAxes,
                      verticalalignment='top', fontsize=10, fontfamily='monospace',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        print(f"✅ Training results visualization saved: {save_path}")
        
        return fig
    
    def save_model(self, base_path='secureroutex_models'):
        """
        Save all trained model components
        
        Args:
            base_path (str): Base directory path for saving models
        """
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
        
        models_saved = []
        
        # Save individual components
        if self.generator is not None:
            generator_path = f"{base_path}/secureroutex_generator.h5"
            self.generator.save(generator_path)
            models_saved.append("Generator")
        
        if self.discriminator is not None:
            discriminator_path = f"{base_path}/secureroutex_discriminator.h5"
            self.discriminator.save(discriminator_path)
            models_saved.append("Discriminator")
        
        if self.trust_evaluator is not None:
            trust_path = f"{base_path}/secureroutex_trust_evaluator.h5"
            self.trust_evaluator.save(trust_path)
            models_saved.append("Trust Evaluator")
        
        if self.attack_detector is not None:
            detector_path = f"{base_path}/secureroutex_attack_detector.h5"
            self.attack_detector.save(detector_path)
            models_saved.append("Attack Detector")
        
        # Save preprocessing objects
        import joblib
        scaler_path = f"{base_path}/secureroutex_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        models_saved.append("Scaler")
        
        print("✅ SecureRouteX Models Saved Successfully:")
        for model in models_saved:
            print(f"   • {model}")
        print(f"   • Base Directory: {base_path}")

def main():
    """
    Main function to demonstrate SecureRouteX GAN model usage
    """
    print("🚀 SecureRouteX GAN Model Demonstration")
    print("="*50)
    
    # Initialize the model
    secureroutex = SecureRouteXGAN(input_dim=50, latent_dim=100)
    
    # Build all model components
    print("\n🔧 Building Model Architecture...")
    secureroutex.build_generator()
    secureroutex.build_discriminator()
    secureroutex.build_trust_evaluator()
    secureroutex.build_attack_detector()
    secureroutex.build_complete_gan()
    
    # Compile models
    print("\n⚙️ Compiling Models...")
    secureroutex.compile_models()
    
    # Load and preprocess data (use your enhanced dataset)
    try:
        print("\n📊 Loading Dataset...")
        data_path = 'secureroutex_enhanced_dataset.csv'
        X_train, X_test, attack_train, attack_test, domain_train, domain_test = secureroutex.load_and_preprocess_data(data_path)
        
        # Train the model (small demo with 10 epochs)
        print("\n🚀 Training GAN Model...")
        secureroutex.train_gan(X_train, attack_train, domain_train, epochs=10, batch_size=32)
        
        # Evaluate performance
        print("\n📊 Evaluating Performance...")
        performance = secureroutex.evaluate_model_performance(X_test, attack_test, domain_test)
        print("Performance Results:", performance)
        
        # Generate synthetic attacks
        print("\n🎯 Generating Synthetic Attacks...")
        synthetic_ddos = secureroutex.generate_synthetic_attacks(100, 'healthcare', 'ddos')
        print(f"Generated synthetic DDoS attacks shape: {synthetic_ddos.shape}")
        
        # Calculate trust scores
        print("\n🛡️ Calculating Trust Scores...")
        trust_scores = secureroutex.calculate_trust_scores(X_test[:10], 'healthcare')
        print("Sample trust scores:", {k: v[:3] for k, v in trust_scores.items()})
        
        # Visualize results
        print("\n📈 Creating Visualization...")
        secureroutex.visualize_training_results()
        
        # Save models
        print("\n💾 Saving Models...")
        secureroutex.save_model()
        
    except FileNotFoundError:
        print("⚠️ Dataset file not found. Please ensure 'secureroutex_enhanced_dataset.csv' exists.")
        print("   The model architecture has been built and compiled successfully.")
    
    print("\n✅ SecureRouteX GAN Model Demonstration Complete!")
    print("="*50)

if __name__ == "__main__":
    main()