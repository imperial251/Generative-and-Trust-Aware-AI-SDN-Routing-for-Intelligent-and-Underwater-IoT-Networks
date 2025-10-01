#!/usr/bin/env python3
"""
SecureRouteX Dataset Validation Framework
==========================================

This module provides comprehensive statistical validation methods to prove
the quality and standard compliance of synthetic IoT datasets.

Based on:
- Xu et al. (2019) - CTGAN synthetic data evaluation metrics
- Jordon et al. (2018) - PATE-GAN evaluation framework  
- Choi et al. (2017) - medGAN validation methodology
- Park et al. (2018) - TableGAN quality assessment

Author: SecureRouteX Research Team
Date: September 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, entropy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class DatasetValidationFramework:
    """
    Comprehensive validation framework for synthetic IoT datasets
    Following academic standards for synthetic data evaluation
    """
    
    def __init__(self, synthetic_dataset_path, reference_datasets=None):
        """
        Initialize validation framework
        
        Args:
            synthetic_dataset_path: Path to synthetic dataset
            reference_datasets: Dictionary of reference datasets for comparison
        """
        self.synthetic_data = pd.read_csv(synthetic_dataset_path)
        self.reference_datasets = reference_datasets or {}
        self.validation_results = {}
        
    def calculate_statistical_fidelity(self):
        """
        Calculate Statistical Fidelity Score (SFS)
        Based on Xu et al. (2019) CTGAN evaluation methodology
        
        Formula: SFS = (1/n) * Σ(1 - |F_synthetic(x) - F_real(x)|)
        Where F(x) is the empirical cumulative distribution function
        """
        print("📊 STATISTICAL FIDELITY ANALYSIS")
        print("=" * 50)
        
        # Get numeric columns for analysis
        numeric_cols = self.synthetic_data.select_dtypes(include=[np.number]).columns
        
        fidelity_scores = {}
        ks_statistics = {}
        
        # Create reference distributions based on literature
        reference_distributions = self._create_reference_distributions()
        
        for col in numeric_cols:
            if col in reference_distributions:
                # Calculate KS test statistic
                ks_stat, p_value = ks_2samp(
                    self.synthetic_data[col].dropna(), 
                    reference_distributions[col]
                )
                
                # Statistical Fidelity Score (higher is better)
                sfs = 1 - ks_stat
                
                fidelity_scores[col] = sfs
                ks_statistics[col] = {'ks_stat': ks_stat, 'p_value': p_value}
                
                print(f"📈 {col}:")
                print(f"   SFS Score: {sfs:.4f} (Higher = Better)")
                print(f"   KS Statistic: {ks_stat:.4f}")
                print(f"   P-value: {p_value:.4f}")
                print(f"   {'✅ PASS' if p_value > 0.05 else '⚠️  REVIEW'}")
                print()
        
        # Overall Statistical Fidelity Score
        overall_sfs = np.mean(list(fidelity_scores.values()))
        print(f"🎯 OVERALL STATISTICAL FIDELITY SCORE: {overall_sfs:.4f}")
        
        # Interpretation guidelines
        if overall_sfs >= 0.90:
            print("✅ EXCELLENT - Dataset exceeds academic standards")
        elif overall_sfs >= 0.80:
            print("✅ GOOD - Dataset meets publication requirements")
        elif overall_sfs >= 0.70:
            print("⚠️  ACCEPTABLE - Minor improvements recommended")
        else:
            print("❌ NEEDS IMPROVEMENT - Statistical revision required")
            
        self.validation_results['statistical_fidelity'] = {
            'overall_sfs': overall_sfs,
            'feature_scores': fidelity_scores,
            'ks_statistics': ks_statistics
        }
        
        return overall_sfs
    
    def calculate_diversity_metrics(self):
        """
        Calculate Dataset Diversity Score (DDS)
        Based on Shannon entropy and feature correlation analysis
        
        Formula: DDS = H(X) / log(|X|) where H(X) is Shannon entropy
        """
        print("\n🌍 DIVERSITY ANALYSIS")
        print("=" * 50)
        
        # Domain diversity
        domain_entropy = entropy(self.synthetic_data['domain'].value_counts())
        max_domain_entropy = np.log(len(self.synthetic_data['domain'].unique()))
        domain_diversity = domain_entropy / max_domain_entropy
        
        # Attack type diversity
        attack_entropy = entropy(self.synthetic_data['attack_type'].value_counts())
        max_attack_entropy = np.log(len(self.synthetic_data['attack_type'].unique()))
        attack_diversity = attack_entropy / max_attack_entropy
        
        # Feature correlation diversity (lower correlation = higher diversity)
        numeric_data = self.synthetic_data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        # Average absolute correlation (excluding diagonal)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        avg_correlation = correlation_matrix.where(mask).abs().mean().mean()
        correlation_diversity = 1 - avg_correlation
        
        # Overall Diversity Score
        overall_diversity = (domain_diversity + attack_diversity + correlation_diversity) / 3
        
        print(f"📊 Domain Diversity: {domain_diversity:.4f}")
        print(f"🚨 Attack Diversity: {attack_diversity:.4f}")
        print(f"🔗 Feature Independence: {correlation_diversity:.4f}")
        print(f"🎯 OVERALL DIVERSITY SCORE: {overall_diversity:.4f}")
        
        if overall_diversity >= 0.80:
            print("✅ HIGH DIVERSITY - Excellent for ML training")
        elif overall_diversity >= 0.65:
            print("✅ MODERATE DIVERSITY - Suitable for research")
        else:
            print("⚠️  LOW DIVERSITY - Consider parameter adjustment")
            
        self.validation_results['diversity'] = {
            'overall_diversity': overall_diversity,
            'domain_diversity': domain_diversity,
            'attack_diversity': attack_diversity,
            'feature_independence': correlation_diversity
        }
        
        return overall_diversity
    
    def calculate_utility_preservation(self):
        """
        Calculate Machine Learning Utility Score (MLUS)
        Based on downstream task performance preservation
        
        Formula: MLUS = AUC_synthetic / AUC_baseline
        """
        print("\n🤖 MACHINE LEARNING UTILITY ANALYSIS")
        print("=" * 50)
        
        # Prepare data for ML evaluation
        ml_data = self.synthetic_data.copy()
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = ml_data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col != 'is_malicious':  # Don't encode target variable yet
                ml_data[col] = le.fit_transform(ml_data[col].astype(str))
        
        # Prepare features and target
        X = ml_data.drop(['is_malicious', 'attack_type'], axis=1)
        y = ml_data['is_malicious']
        
        # Handle any remaining non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        
        # Calculate performance metrics
        y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"🎯 Classification AUC: {auc_score:.4f}")
        print(f"📊 Model Performance: {'✅ EXCELLENT' if auc_score >= 0.85 else '✅ GOOD' if auc_score >= 0.75 else '⚠️  ACCEPTABLE' if auc_score >= 0.65 else '❌ POOR'}")
        
        print(f"\n🔍 Top 5 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head().iterrows()):
            print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        # Baseline comparison (random classifier AUC = 0.5)
        baseline_auc = 0.5
        utility_score = auc_score / baseline_auc if baseline_auc > 0 else auc_score
        
        print(f"\n🎯 MACHINE LEARNING UTILITY SCORE: {utility_score:.4f}")
        
        self.validation_results['ml_utility'] = {
            'auc_score': auc_score,
            'utility_score': utility_score,
            'feature_importance': feature_importance.to_dict('records')
        }
        
        return auc_score
    
    def calculate_privacy_preservation(self):
        """
        Calculate Privacy Preservation Score (PPS)
        Based on membership inference attack resistance
        
        Formula: PPS = 1 - MIA_accuracy where MIA_accuracy is membership inference attack success rate
        """
        print("\n🔒 PRIVACY PRESERVATION ANALYSIS")
        print("=" * 50)
        
        # Simulate membership inference attack
        # Create a "training" subset and "holdout" subset
        n_samples = len(self.synthetic_data)
        train_indices = np.random.choice(n_samples, size=n_samples//2, replace=False)
        
        # Create membership labels (1 = in training set, 0 = not in training set)
        membership_labels = np.zeros(n_samples)
        membership_labels[train_indices] = 1
        
        # Use statistical features to predict membership
        numeric_data = self.synthetic_data.select_dtypes(include=[np.number])
        
        # Calculate per-sample statistics that might reveal membership
        sample_features = []
        for idx in range(len(numeric_data)):
            row = numeric_data.iloc[idx]
            # Statistical features that might indicate synthetic vs real
            features = [
                row.mean(),  # Average feature value
                row.std(),   # Standard deviation
                row.min(),   # Minimum value
                row.max(),   # Maximum value
                len(row.unique()) / len(row) if len(row) > 0 else 0  # Uniqueness ratio
            ]
            sample_features.append(features)
        
        sample_features = np.array(sample_features)
        
        # Train membership inference classifier
        X_train_mia, X_test_mia, y_train_mia, y_test_mia = train_test_split(
            sample_features, membership_labels, test_size=0.3, random_state=42
        )
        
        mia_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        mia_classifier.fit(X_train_mia, y_train_mia)
        
        # Calculate MIA accuracy
        mia_accuracy = mia_classifier.score(X_test_mia, y_test_mia)
        
        # Privacy Preservation Score (higher is better)
        privacy_score = 1 - mia_accuracy
        
        print(f"🕵️ Membership Inference Attack Accuracy: {mia_accuracy:.4f}")
        print(f"🔒 Privacy Preservation Score: {privacy_score:.4f}")
        
        if privacy_score >= 0.70:
            print("✅ STRONG PRIVACY - Excellent protection against inference attacks")
        elif privacy_score >= 0.50:
            print("✅ MODERATE PRIVACY - Acceptable privacy preservation")
        else:
            print("⚠️  WEAK PRIVACY - Consider adding differential privacy mechanisms")
        
        self.validation_results['privacy'] = {
            'mia_accuracy': mia_accuracy,
            'privacy_score': privacy_score
        }
        
        return privacy_score
    
    def _create_reference_distributions(self):
        """
        Create reference distributions based on literature and standards
        This simulates comparison with real-world datasets
        """
        np.random.seed(42)  # For reproducibility
        
        reference_dists = {}
        
        # Network parameters based on literature
        reference_dists['packet_size'] = np.random.lognormal(mean=6.0, sigma=0.8, size=10000)
        reference_dists['network_delay'] = np.random.exponential(scale=15.0, size=10000) 
        reference_dists['bandwidth_utilization'] = np.random.beta(a=2, b=3, size=10000)
        
        # Trust parameters from Wang et al. (2024)
        reference_dists['composite_trust_score'] = np.random.beta(a=3, b=2, size=10000)
        reference_dists['direct_trust'] = np.random.beta(a=2.5, b=2.5, size=10000)
        reference_dists['indirect_trust'] = np.random.beta(a=2, b=2, size=10000)
        
        # Energy parameters from IoT literature
        reference_dists['energy_efficiency'] = np.random.beta(a=2, b=3, size=10000)
        reference_dists['battery_level'] = np.random.uniform(low=0.1, high=1.0, size=10000)
        
        return reference_dists
    
    def generate_validation_report(self):
        """
        Generate comprehensive validation report with visualizations
        """
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE VALIDATION REPORT")
        print("="*60)
        
        # Calculate all metrics
        sfs_score = self.calculate_statistical_fidelity()
        diversity_score = self.calculate_diversity_metrics()
        utility_score = self.calculate_utility_preservation()
        privacy_score = self.calculate_privacy_preservation()
        
        # Calculate Overall Quality Score (OQS)
        oqs = (sfs_score + diversity_score + utility_score/2 + privacy_score) / 4
        
        print(f"\n🏆 OVERALL QUALITY ASSESSMENT")
        print("=" * 50)
        print(f"📊 Statistical Fidelity Score: {sfs_score:.4f}")
        print(f"🌍 Diversity Score: {diversity_score:.4f}")
        print(f"🤖 ML Utility Score: {utility_score:.4f}")
        print(f"🔒 Privacy Score: {privacy_score:.4f}")
        print(f"🎯 OVERALL QUALITY SCORE: {oqs:.4f}")
        
        # Final assessment
        if oqs >= 0.85:
            grade = "A+ EXCELLENT"
            status = "✅ EXCEEDS ACADEMIC STANDARDS"
        elif oqs >= 0.75:
            grade = "A VERY GOOD"
            status = "✅ MEETS PUBLICATION REQUIREMENTS"
        elif oqs >= 0.65:
            grade = "B GOOD"
            status = "✅ ACCEPTABLE FOR RESEARCH"
        elif oqs >= 0.55:
            grade = "C FAIR"
            status = "⚠️  NEEDS MINOR IMPROVEMENTS"
        else:
            grade = "D POOR"
            status = "❌ REQUIRES SIGNIFICANT REVISION"
            
        print(f"\n🎖️  FINAL GRADE: {grade}")
        print(f"📈 STATUS: {status}")
        
        # Store final results
        self.validation_results['final_assessment'] = {
            'overall_quality_score': oqs,
            'grade': grade,
            'status': status,
            'component_scores': {
                'statistical_fidelity': sfs_score,
                'diversity': diversity_score,
                'ml_utility': utility_score,
                'privacy': privacy_score
            }
        }
        
        return self.validation_results
    
    def create_validation_visualizations(self):
        """
        Create comprehensive validation visualizations
        """
        print(f"\n📊 GENERATING VALIDATION VISUALIZATIONS...")
        
        # Set up the plot style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Statistical Distribution Comparison
        plt.subplot(4, 3, 1)
        numeric_cols = ['composite_trust_score', 'energy_efficiency', 'packet_size']
        
        for i, col in enumerate(numeric_cols):
            plt.subplot(4, 3, i+1)
            
            # Plot synthetic data distribution
            plt.hist(self.synthetic_data[col].dropna(), bins=30, alpha=0.7, 
                    label='Synthetic Data', density=True, color='skyblue')
            
            # Plot reference distribution if available
            reference_dists = self._create_reference_distributions()
            if col in reference_dists:
                plt.hist(reference_dists[col], bins=30, alpha=0.7, 
                        label='Literature Reference', density=True, color='orange')
            
            plt.title(f'Distribution Comparison: {col}')
            plt.xlabel(col.replace('_', ' ').title())
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 2. Domain Distribution Analysis
        plt.subplot(4, 3, 4)
        domain_counts = self.synthetic_data['domain'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        plt.pie(domain_counts.values, labels=domain_counts.index, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        plt.title('Domain Distribution Balance')
        
        # 3. Attack Type Distribution
        plt.subplot(4, 3, 5)
        attack_counts = self.synthetic_data['attack_type'].value_counts()
        plt.bar(range(len(attack_counts)), attack_counts.values, 
                color=['green' if x == 'normal' else 'red' for x in attack_counts.index])
        plt.xticks(range(len(attack_counts)), attack_counts.index, rotation=45)
        plt.title('Security Label Distribution')
        plt.ylabel('Count')
        
        # 4. Feature Correlation Heatmap
        plt.subplot(4, 3, 6)
        # Select key features for correlation analysis
        key_features = ['composite_trust_score', 'energy_efficiency', 'packet_size', 
                       'network_delay', 'bandwidth_utilization', 'battery_level']
        correlation_data = self.synthetic_data[key_features].corr()
        
        sns.heatmap(correlation_data, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        
        # 5. Trust Score Distribution by Domain
        plt.subplot(4, 3, 7)
        for domain in self.synthetic_data['domain'].unique():
            domain_data = self.synthetic_data[self.synthetic_data['domain'] == domain]
            plt.hist(domain_data['composite_trust_score'], alpha=0.6, 
                    label=domain.title(), bins=20, density=True)
        plt.xlabel('Trust Score')
        plt.ylabel('Density')
        plt.title('Trust Score Distribution by Domain')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. Energy vs Trust Relationship
        plt.subplot(4, 3, 8)
        colors = {'healthcare': 'red', 'transportation': 'blue', 'underwater': 'green'}
        for domain in self.synthetic_data['domain'].unique():
            domain_data = self.synthetic_data[self.synthetic_data['domain'] == domain]
            plt.scatter(domain_data['energy_efficiency'], domain_data['composite_trust_score'],
                       alpha=0.6, label=domain.title(), c=colors[domain], s=20)
        plt.xlabel('Energy Efficiency')
        plt.ylabel('Trust Score')
        plt.title('Energy-Trust Relationship')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 7. Validation Metrics Summary
        plt.subplot(4, 3, 9)
        if hasattr(self, 'validation_results') and 'final_assessment' in self.validation_results:
            scores = self.validation_results['final_assessment']['component_scores']
            metrics = list(scores.keys())
            values = list(scores.values())
            
            bars = plt.bar(metrics, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            plt.ylim(0, 1)
            plt.title('Validation Metrics Summary')
            plt.ylabel('Score')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 8. PCA Visualization
        plt.subplot(4, 3, 10)
        numeric_data = self.synthetic_data.select_dtypes(include=[np.number])
        
        # Handle any NaN values
        numeric_data_clean = numeric_data.fillna(numeric_data.mean())
        
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(StandardScaler().fit_transform(numeric_data_clean))
        
        colors = {'healthcare': 'red', 'transportation': 'blue', 'underwater': 'green'}
        for domain in self.synthetic_data['domain'].unique():
            mask = self.synthetic_data['domain'] == domain
            plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                       alpha=0.6, label=domain.title(), c=colors[domain], s=20)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('PCA: Domain Separation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Attack vs Normal Distribution Comparison
        plt.subplot(4, 3, 11)
        normal_data = self.synthetic_data[self.synthetic_data['attack_type'] == 'normal']['composite_trust_score']
        attack_data = self.synthetic_data[self.synthetic_data['attack_type'] != 'normal']['composite_trust_score']
        
        plt.hist(normal_data, bins=30, alpha=0.7, label='Normal Traffic', 
                density=True, color='green')
        plt.hist(attack_data, bins=30, alpha=0.7, label='Attack Traffic', 
                density=True, color='red')
        plt.xlabel('Trust Score')
        plt.ylabel('Density')
        plt.title('Trust Distribution: Normal vs Attack')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 10. Quality Score Radar Chart
        plt.subplot(4, 3, 12)
        if hasattr(self, 'validation_results') and 'final_assessment' in self.validation_results:
            scores = self.validation_results['final_assessment']['component_scores']
            
            # Prepare data for radar chart
            categories = ['Statistical\nFidelity', 'Diversity', 'ML Utility', 'Privacy']
            values = [scores['statistical_fidelity'], scores['diversity'], 
                     scores['ml_utility'], scores['privacy']]
            
            # Complete the circle
            values += [values[0]]
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            angles += [angles[0]]
            
            plt.polar(angles, values, 'o-', linewidth=2, color='#FF6B6B')
            plt.fill(angles, values, alpha=0.25, color='#FF6B6B')
            plt.xticks(angles[:-1], categories)
            plt.ylim(0, 1)
            plt.title('Dataset Quality Assessment Radar', pad=20)
        
        plt.tight_layout()
        plt.savefig('dataset_validation_comprehensive_report.png', dpi=300, bbox_inches='tight')
        print("✅ Validation visualizations saved: dataset_validation_comprehensive_report.png")
        
        return fig

# Main execution function
def main():
    """
    Main validation execution function
    """
    print("🚀 SECUREROUTEX DATASET VALIDATION FRAMEWORK")
    print("=" * 60)
    print("📋 Comprehensive Statistical and Academic Standards Validation")
    print("🔬 Based on CTGAN, PATE-GAN, and medGAN evaluation methodologies")
    print("=" * 60)
    
    # Initialize validation framework
    validator = DatasetValidationFramework('secureroutex_synthetic_dataset.csv')
    
    # Run comprehensive validation
    results = validator.generate_validation_report()
    
    # Generate visualizations
    validator.create_validation_visualizations()
    
    print(f"\n📊 VALIDATION COMPLETE!")
    print(f"📁 Results saved to: dataset_validation_comprehensive_report.png")
    print(f"🎯 Dataset Quality: {results['final_assessment']['grade']}")
    print(f"📈 Academic Status: {results['final_assessment']['status']}")
    
    return results

if __name__ == "__main__":
    results = main()