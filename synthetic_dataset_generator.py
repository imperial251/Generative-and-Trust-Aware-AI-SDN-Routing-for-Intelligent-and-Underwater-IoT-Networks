"""
Synthetic Dataset Generator for Generative and Trust-Aware AI-SDN Routing
Based on comprehensive analysis of reference papers including:
- Zouhri et al. (2025) - CTGAN-ENN Framework
- Wang et al. (2024) - GTR Underwater Networks  
- Anantula et al. (2025) - Cloud IoT Security
- Khan et al. (2024) - Healthcare IoT Routing
- Song et al. (2025) - Emergency ITS Routing

Author: Research Team
Date: September 2025
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SecureRouteXDatasetGenerator:
    """
    Comprehensive dataset generator for multi-domain IoT networks
    Following methodologies from referenced papers
    """
    
    def __init__(self, seed=42):
        """
        Initialize generator with domain-specific configurations
        
        Parameters based on literature analysis:
        - Healthcare IoT: Khan et al. (2024) - 12% throughput improvement, 20% latency reduction
        - Transportation: Song et al. (2025) - 95% accuracy, 105ms latency
        - Underwater: Wang et al. (2024) - 11% packet delivery improvement, 20.4% throughput
        """
        np.random.seed(seed)
        
        # Domain configurations extracted from reference papers
        self.domain_configs = {
            'healthcare': {
                # Based on Khan et al. (2024) and Lv et al. (2025)
                'packet_size': {'mean': 256, 'std': 64, 'min': 64, 'max': 1500},
                'inter_arrival': {'lambda': 0.5},  # Healthcare requires frequent monitoring
                'energy_factor': 0.1,  # Low energy devices (sensors, wearables)
                'criticality_levels': [1, 2, 3, 4, 5],  # Patient criticality
                'base_trust': 0.85,  # High trust requirement for healthcare
                'qos_priority': 'high'  # Critical for patient safety
            },
            'transportation': {
                # Based on Song et al. (2025) and Almadhor et al. (2025)
                'packet_size': {'mean': 512, 'std': 128, 'min': 128, 'max': 2048},
                'inter_arrival': {'lambda': 0.1},  # High frequency for real-time updates
                'energy_factor': 0.08,  # Vehicle-powered, less energy constraint
                'velocity_range': [0, 120],  # Vehicle speed in km/h
                'base_trust': 0.75,  # Moderate trust in vehicular networks
                'qos_priority': 'medium'
            },
            'underwater': {
                # Based on Wang et al. (2024) GTR algorithm
                'packet_size': {'mean': 128, 'std': 32, 'min': 32, 'max': 512},
                'inter_arrival': {'lambda': 2.0},  # Lower frequency due to propagation
                'energy_factor': 0.25,  # High energy cost for underwater transmission
                'depth_range': [10, 1000],  # Underwater depth in meters
                'base_trust': 0.65,  # Lower base trust due to harsh environment
                'qos_priority': 'low'  # Delay-tolerant applications
            }
        }
        
        # Attack patterns based on Anantula et al. (2025) and Zouhri et al. (2025)
        self.attack_patterns = {
            'ddos': {
                'packet_size_factor': 0.5,  # Smaller packets
                'frequency_factor': 10,     # Much higher frequency
                'energy_impact': 2.0,       # Higher energy consumption
                'trust_degradation': 0.6    # Significant trust reduction
            },
            'malicious_node': {
                'packet_size_factor': 1.2,  # Irregular packet sizes
                'frequency_factor': 0.3,    # Irregular timing
                'energy_impact': 1.5,       # Moderate energy impact
                'trust_degradation': 0.8    # Gradual trust degradation
            },
            'energy_drain': {
                'packet_size_factor': 3.0,  # Large packets to drain energy
                'frequency_factor': 2.0,    # Frequent large packets
                'energy_impact': 5.0,       # Severe energy impact
                'trust_degradation': 0.4    # Rapid trust degradation
            },
            'routing_attack': {
                'packet_size_factor': 1.0,  # Normal packet size
                'frequency_factor': 1.5,    # Slightly higher frequency
                'energy_impact': 1.2,       # Minimal energy impact
                'trust_degradation': 0.7    # Moderate trust impact
            }
        }
    
    def generate_network_features(self, domain, n_samples, is_attack=False, attack_type=None):
        """
        Generate network-level features based on Zouhri et al. (2025) CTGAN approach
        
        Args:
            domain: IoT domain ('healthcare', 'transportation', 'underwater')
            n_samples: Number of samples to generate
            is_attack: Whether to generate attack traffic
            attack_type: Type of attack if is_attack=True
        
        Returns:
            DataFrame with network features
        """
        config = self.domain_configs[domain]
        
        # Base packet characteristics
        if is_attack and attack_type:
            attack_config = self.attack_patterns[attack_type]
            # Modify packet size based on attack pattern
            packet_size_mean = config['packet_size']['mean'] * attack_config['packet_size_factor']
            inter_arrival_lambda = config['inter_arrival']['lambda'] / attack_config['frequency_factor']
        else:
            packet_size_mean = config['packet_size']['mean']
            inter_arrival_lambda = config['inter_arrival']['lambda']
        
        # Packet size distribution (following Zouhri et al. approach)
        packet_sizes = np.random.gamma(
            shape=2, 
            scale=packet_size_mean/2, 
            size=n_samples
        ).astype(int)
        packet_sizes = np.clip(packet_sizes, config['packet_size']['min'], config['packet_size']['max'])
        
        # Inter-arrival time (exponential distribution as per literature)
        inter_arrival_times = np.random.exponential(inter_arrival_lambda, n_samples)
        
        # Flow duration (based on domain characteristics)
        if domain == 'healthcare':
            # Longer sessions for continuous monitoring
            flow_duration = np.random.gamma(shape=3, scale=60, size=n_samples)
        elif domain == 'transportation':
            # Variable duration based on trip length
            flow_duration = np.random.gamma(shape=2, scale=30, size=n_samples)
        else:  # underwater
            # Shorter bursts due to energy constraints
            flow_duration = np.random.gamma(shape=1.5, scale=20, size=n_samples)
        
        # Network delay (including domain-specific propagation characteristics)
        if domain == 'underwater':
            # Underwater acoustic propagation delay (Wang et al. 2024)
            base_delay = np.random.gamma(shape=3, scale=50, size=n_samples)
        else:
            base_delay = np.random.gamma(shape=2, scale=10, size=n_samples)
        
        # Add attack-specific delay if applicable
        if is_attack:
            delay_multiplier = self.attack_patterns[attack_type]['energy_impact']
            network_delay = base_delay * delay_multiplier
        else:
            network_delay = base_delay
        
        return pd.DataFrame({
            'packet_size': packet_sizes,
            'inter_arrival_time': inter_arrival_times,
            'flow_duration': flow_duration,
            'network_delay': network_delay,
            'bandwidth_utilization': np.random.beta(2, 5, n_samples),  # Typically low utilization
            'protocol_type': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples, p=[0.6, 0.35, 0.05])
        })
    
    def generate_trust_features(self, domain, n_samples, is_attack=False, attack_type=None):
        """
        Generate trust-based features following Wang et al. (2024) GTR methodology
        
        Trust evaluation components:
        1. Energy Trust: Based on remaining energy levels
        2. Direct Trust: Based on direct interaction history  
        3. Indirect Trust: Based on neighbor recommendations
        """
        config = self.domain_configs[domain]
        base_trust = config['base_trust']
        
        if is_attack and attack_type:
            trust_degradation = self.attack_patterns[attack_type]['trust_degradation']
            base_trust *= trust_degradation
        
        # Energy Trust (Wang et al. 2024 - energy levels crucial for trust)
        if domain == 'underwater':
            # Underwater nodes have limited energy (higher variability)
            energy_levels = np.random.beta(3, 7, n_samples)  # Lower energy distribution
        elif domain == 'healthcare':
            # Healthcare devices need reliable power
            energy_levels = np.random.beta(8, 2, n_samples)  # Higher energy distribution
        else:  # transportation
            # Vehicle-powered, generally good energy
            energy_levels = np.random.beta(6, 3, n_samples)
        
        # Direct Trust (based on successful packet delivery)
        if is_attack:
            # Attacks reduce packet success rate
            packet_success_rate = np.random.beta(2, 8, n_samples)  # Lower success
        else:
            packet_success_rate = np.random.beta(9, 1, n_samples)  # High success
        
        # Indirect Trust (neighbor recommendations)
        # Following Wang et al. approach of neighbor-based trust
        neighbor_trust = np.random.beta(
            int(base_trust * 10), 
            int((1 - base_trust) * 10), 
            n_samples
        )
        
        # Response Time Trust (lower is better)
        if is_attack:
            response_times = np.random.gamma(5, 20, n_samples)  # Higher response times
        else:
            response_times = np.random.gamma(2, 5, n_samples)   # Lower response times
        
        # Composite trust score (GTR algorithm approach)
        composite_trust = (
            0.3 * energy_levels + 
            0.4 * packet_success_rate + 
            0.2 * neighbor_trust + 
            0.1 * (1 / (1 + response_times/100))  # Inverse relationship with response time
        )
        
        return pd.DataFrame({
            'energy_trust': energy_levels,
            'direct_trust': packet_success_rate,
            'indirect_trust': neighbor_trust,
            'response_time': response_times,
            'composite_trust_score': composite_trust,
            'trust_history_length': np.random.poisson(lam=50, size=n_samples),
            'trust_variance': np.random.gamma(1, 0.1, n_samples)
        })
    
    def generate_domain_specific_features(self, domain, n_samples, is_attack=False):
        """
        Generate domain-specific features based on literature analysis
        """
        features = {}
        
        if domain == 'healthcare':
            # Based on Khan et al. (2024) and Lv et al. (2025)
            features.update({
                'patient_criticality': np.random.choice([1, 2, 3, 4, 5], n_samples, 
                                                       p=[0.4, 0.3, 0.15, 0.1, 0.05]),
                'device_type': np.random.choice(['sensor', 'monitor', 'pump', 'ventilator'], 
                                              n_samples, p=[0.4, 0.3, 0.2, 0.1]),
                'data_sensitivity': np.random.choice(['low', 'medium', 'high'], n_samples,
                                                   p=[0.3, 0.4, 0.3]),
                'real_time_requirement': np.random.beta(8, 2, n_samples),  # High real-time needs
                'encryption_level': np.random.choice([128, 256, 512], n_samples, p=[0.2, 0.6, 0.2])
            })
        
        elif domain == 'transportation':
            # Based on Song et al. (2025) and Almadhor et al. (2025)
            features.update({
                'vehicle_speed': np.random.gamma(3, 15, n_samples),  # km/h
                'location_accuracy': np.random.beta(7, 3, n_samples),
                'traffic_density': np.random.choice(['low', 'medium', 'high'], n_samples,
                                                  p=[0.4, 0.4, 0.2]),
                'emergency_level': np.random.choice([0, 1, 2, 3], n_samples, p=[0.85, 0.1, 0.03, 0.02]),
                'weather_condition': np.random.choice(['clear', 'rain', 'fog', 'snow'], n_samples,
                                                    p=[0.6, 0.25, 0.1, 0.05]),
                'road_type': np.random.choice(['highway', 'city', 'rural'], n_samples, p=[0.3, 0.5, 0.2])
            })
        
        else:  # underwater
            # Based on Wang et al. (2024) GTR methodology
            features.update({
                'depth': np.random.uniform(10, 1000, n_samples),
                'water_temperature': np.random.normal(15, 5, n_samples),
                'salinity': np.random.normal(35, 2, n_samples),  # PSU
                'current_speed': np.random.gamma(1, 0.5, n_samples),  # m/s
                'acoustic_noise': np.random.gamma(2, 10, n_samples),  # dB
                'signal_attenuation': np.random.beta(3, 2, n_samples),
                'node_mobility': np.random.choice(['static', 'slow', 'medium'], n_samples,
                                                p=[0.6, 0.3, 0.1])
            })
        
        return pd.DataFrame(features)
    
    def generate_sdn_features(self, domain, n_samples, is_attack=False, attack_type=None):
        """
        Generate SDN-specific features based on multiple papers
        (Souza et al., Li et al., Byakodi et al.)
        """
        # Controller response metrics
        if is_attack:
            controller_response = np.random.gamma(3, 20, n_samples)  # Higher under attack
            flow_setup_time = np.random.gamma(4, 15, n_samples)
        else:
            controller_response = np.random.gamma(2, 5, n_samples)   # Normal response
            flow_setup_time = np.random.gamma(2, 3, n_samples)
        
        # Flow table characteristics
        flow_table_utilization = np.random.beta(2, 8, n_samples) if not is_attack else np.random.beta(7, 3, n_samples)
        
        # Control channel metrics
        control_overhead = np.random.gamma(1, 2, n_samples)
        if is_attack and attack_type == 'ddos':
            control_overhead *= 5  # DDoS increases control overhead significantly
        
        return pd.DataFrame({
            'controller_response_time': controller_response,
            'flow_setup_time': flow_setup_time,
            'flow_table_utilization': flow_table_utilization,
            'control_channel_overhead': control_overhead,
            'switch_cpu_utilization': np.random.beta(3, 7, n_samples),
            'rule_installation_latency': np.random.gamma(1, 2, n_samples)
        })
    
    def generate_energy_features(self, domain, n_samples, is_attack=False, attack_type=None):
        """
        Generate energy-related features following Fu et al. (2024) methodology
        """
        config = self.domain_configs[domain]
        base_energy_factor = config['energy_factor']
        
        if is_attack and attack_type:
            energy_multiplier = self.attack_patterns[attack_type]['energy_impact']
        else:
            energy_multiplier = 1.0
        
        # Transmission energy (dominant component)
        transmission_energy = np.random.gamma(2, base_energy_factor * 10, n_samples) * energy_multiplier
        
        # Processing energy
        processing_energy = np.random.gamma(1.5, base_energy_factor * 5, n_samples)
        
        # Idle energy consumption
        idle_energy = np.random.gamma(1, base_energy_factor * 2, n_samples)
        
        # Total energy consumption
        total_energy = transmission_energy + processing_energy + idle_energy
        
        # Remaining battery (important for trust calculation)
        if domain == 'underwater':
            battery_level = np.random.beta(3, 7, n_samples)  # Generally lower
        else:
            battery_level = np.random.beta(6, 4, n_samples)  # Generally higher
        
        return pd.DataFrame({
            'transmission_energy': transmission_energy,
            'processing_energy': processing_energy,
            'idle_energy': idle_energy,
            'total_energy_consumption': total_energy,
            'battery_level': battery_level,
            'energy_efficiency': np.random.beta(5, 3, n_samples)
        })
    
    def generate_complete_dataset(self, domain_samples=None, attack_ratio=0.2):
        """
        Generate complete dataset for all domains with proper balance
        
        Args:
            domain_samples: Dict with samples per domain {'healthcare': 5000, 'transportation': 5000, 'underwater': 5000}
            attack_ratio: Proportion of attack samples (default 0.2 = 20% attacks)
        """
        if domain_samples is None:
            domain_samples = {'healthcare': 5000, 'transportation': 5000, 'underwater': 5000}
        
        all_datasets = []
        
        for domain, total_samples in domain_samples.items():
            print(f"Generating {total_samples} samples for {domain} domain...")
            
            # Calculate normal vs attack samples
            attack_samples = int(total_samples * attack_ratio)
            normal_samples = total_samples - attack_samples
            
            # Generate normal traffic
            normal_data = self._generate_domain_dataset(domain, normal_samples, is_attack=False)
            normal_data['is_malicious'] = 0
            normal_data['attack_type'] = 'normal'
            
            # Generate attack traffic (distributed across attack types)
            attack_types = list(self.attack_patterns.keys())
            samples_per_attack = attack_samples // len(attack_types)
            
            attack_datasets = []
            for attack_type in attack_types:
                attack_data = self._generate_domain_dataset(
                    domain, samples_per_attack, 
                    is_attack=True, attack_type=attack_type
                )
                attack_data['is_malicious'] = 1
                attack_data['attack_type'] = attack_type
                attack_datasets.append(attack_data)
            
            # Combine normal and attack data
            domain_data = pd.concat([normal_data] + attack_datasets, ignore_index=True)
            domain_data['domain'] = domain
            
            all_datasets.append(domain_data)
        
        # Combine all domains
        complete_dataset = pd.concat(all_datasets, ignore_index=True)
        
        # Shuffle the dataset
        complete_dataset = complete_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Add temporal features
        complete_dataset = self._add_temporal_features(complete_dataset)
        
        print(f"\nDataset generation complete!")
        print(f"Total samples: {len(complete_dataset)}")
        print(f"Domains: {complete_dataset['domain'].value_counts().to_dict()}")
        print(f"Attack distribution: {complete_dataset['attack_type'].value_counts().to_dict()}")
        
        return complete_dataset
    
    def _generate_domain_dataset(self, domain, n_samples, is_attack=False, attack_type=None):
        """Generate complete dataset for a specific domain"""
        
        # Generate all feature categories
        network_features = self.generate_network_features(domain, n_samples, is_attack, attack_type)
        trust_features = self.generate_trust_features(domain, n_samples, is_attack, attack_type)
        domain_features = self.generate_domain_specific_features(domain, n_samples, is_attack)
        sdn_features = self.generate_sdn_features(domain, n_samples, is_attack, attack_type)
        energy_features = self.generate_energy_features(domain, n_samples, is_attack, attack_type)
        
        # Combine all features
        combined_data = pd.concat([
            network_features, trust_features, domain_features, 
            sdn_features, energy_features
        ], axis=1)
        
        return combined_data
    
    def _add_temporal_features(self, dataset):
        """Add temporal features for time-series analysis"""
        
        # Generate timestamps
        start_time = datetime.now()
        timestamps = [start_time + timedelta(seconds=i*np.random.exponential(1)) 
                     for i in range(len(dataset))]
        
        dataset['timestamp'] = timestamps
        dataset['hour'] = dataset['timestamp'].dt.hour
        dataset['day_of_week'] = dataset['timestamp'].dt.dayofweek
        dataset['is_weekend'] = dataset['day_of_week'].isin([5, 6]).astype(int)
        
        return dataset
    
    def validate_dataset_quality(self, dataset):
        """
        Validate the generated dataset quality
        Following best practices from Zouhri et al. (2025)
        """
        print("Dataset Quality Validation Report")
        print("=" * 50)
        
        # 1. Class Balance
        print("\n1. Class Balance Analysis:")
        class_dist = dataset['is_malicious'].value_counts(normalize=True)
        print(f"   Normal: {class_dist[0]:.2%}")
        print(f"   Malicious: {class_dist[1]:.2%}")
        
        # 2. Domain Balance  
        print("\n2. Domain Distribution:")
        domain_dist = dataset['domain'].value_counts(normalize=True)
        for domain, percentage in domain_dist.items():
            print(f"   {domain.capitalize()}: {percentage:.2%}")
        
        # 3. Attack Type Distribution
        print("\n3. Attack Type Distribution:")
        attack_dist = dataset['attack_type'].value_counts(normalize=True)
        for attack, percentage in attack_dist.items():
            print(f"   {attack}: {percentage:.2%}")
        
        # 4. Feature Statistics
        print("\n4. Feature Quality Metrics:")
        numeric_features = dataset.select_dtypes(include=[np.number])
        
        # Check for missing values
        missing_values = numeric_features.isnull().sum().sum()
        print(f"   Missing values: {missing_values}")
        
        # Check for infinite values
        infinite_values = np.isinf(numeric_features).sum().sum()
        print(f"   Infinite values: {infinite_values}")
        
        # Feature correlations
        correlation_matrix = numeric_features.corr()
        high_correlations = (correlation_matrix.abs() > 0.9) & (correlation_matrix != 1.0)
        high_corr_count = high_correlations.sum().sum() // 2
        print(f"   High correlations (>0.9): {high_corr_count}")
        
        # 5. Statistical Validation
        print("\n5. Statistical Properties:")
        print(f"   Mean trust score: {dataset['composite_trust_score'].mean():.3f}")
        print(f"   Mean energy consumption: {dataset['total_energy_consumption'].mean():.3f}")
        print(f"   Mean network delay: {dataset['network_delay'].mean():.3f}ms")
        
        return {
            'class_balance': class_dist,
            'domain_distribution': domain_dist,
            'attack_distribution': attack_dist,
            'correlation_matrix': correlation_matrix,
            'quality_score': self._calculate_quality_score(dataset)
        }
    
    def _calculate_quality_score(self, dataset):
        """Calculate overall dataset quality score"""
        
        # Balance score (closer to 50-50 is better for binary classification)
        class_balance = dataset['is_malicious'].value_counts(normalize=True)
        balance_score = 1 - abs(class_balance[0] - 0.8)  # We want 80-20 split
        
        # Diversity score (domain distribution)
        domain_balance = dataset['domain'].value_counts(normalize=True)
        domain_entropy = stats.entropy(domain_balance)
        max_entropy = np.log(len(domain_balance))
        diversity_score = domain_entropy / max_entropy
        
        # Feature quality score
        numeric_features = dataset.select_dtypes(include=[np.number])
        missing_ratio = numeric_features.isnull().sum().sum() / (len(dataset) * len(numeric_features.columns))
        quality_score = 1 - missing_ratio
        
        overall_score = (balance_score + diversity_score + quality_score) / 3
        
        return {
            'overall': overall_score,
            'balance': balance_score,
            'diversity': diversity_score,  
            'quality': quality_score
        }

# Usage Example and Testing
if __name__ == "__main__":
    print("SecureRouteX Dataset Generator")
    print("Based on comprehensive literature analysis")
    print("=" * 50)
    
    # Initialize generator
    generator = SecureRouteXDatasetGenerator(seed=42)
    
    # Generate dataset
    dataset = generator.generate_complete_dataset(
        domain_samples={'healthcare': 3000, 'transportation': 3000, 'underwater': 3000},
        attack_ratio=0.2
    )
    
    # Validate quality
    validation_report = generator.validate_dataset_quality(dataset)
    
    # Save dataset
    dataset.to_csv('secureroutex_synthetic_dataset.csv', index=False)
    print(f"\nDataset saved as 'secureroutex_synthetic_dataset.csv'")
    print(f"Dataset shape: {dataset.shape}")
    print(f"Quality score: {validation_report['quality_score']['overall']:.3f}")