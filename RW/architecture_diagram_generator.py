#!/usr/bin/env python3
"""
SecureRouteX System Architecture Diagram Generator
=================================================

Generates a comprehensive architecture diagram showing the GAN-based trust-aware 
AI-SDN routing system for heterogeneous IoT networks.

Based on:
- Wang et al. (2024) GTR architecture
- Khan et al. (2024) AI-SDN healthcare routing
- Zabeehullah (2025) SDN-based IoT framework

Author: SecureRouteX Research Team
Date: September 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_secureroutex_architecture():
    """
    Create comprehensive SecureRouteX system architecture diagram
    """
    
    # Create figure with high DPI for publication quality
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'cloud': '#E3F2FD',          # Light blue
        'sdn': '#C8E6C9',            # Light green
        'gan': '#FFE0B2',            # Light orange
        'trust': '#F8BBD9',          # Light pink
        'healthcare': '#FFCDD2',     # Light red
        'transport': '#DCEDC8',      # Light lime
        'underwater': '#B3E5FC',     # Light cyan
        'security': '#D1C4E9',       # Light purple
        'arrow': '#424242',          # Dark gray
        'text': '#212121'            # Very dark gray
    }
    
    # Title
    fig.suptitle('SecureRouteX: Generative and Trust-Aware AI-SDN Routing Architecture\n' + 
                 'for Intelligent and Underwater IoT Networks', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # Layer 1: Cloud Infrastructure & Management Layer (Top)
    cloud_rect = FancyBboxPatch((1, 10), 14, 1.5, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['cloud'], 
                               edgecolor='darkblue', linewidth=2)
    ax.add_patch(cloud_rect)
    
    ax.text(8, 10.75, 'Cloud Infrastructure & Management Layer', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Cloud components
    ax.text(3, 10.4, '• Global Network View\n• Policy Management\n• Analytics Dashboard', 
            ha='left', va='center', fontsize=10)
    ax.text(10, 10.4, '• Cross-Domain Coordination\n• Resource Allocation\n• Performance Monitoring', 
            ha='left', va='center', fontsize=10)
    
    # Layer 2: AI-SDN Control Plane (Upper Middle)
    sdn_rect = FancyBboxPatch((1, 8), 14, 1.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['sdn'], 
                             edgecolor='darkgreen', linewidth=2)
    ax.add_patch(sdn_rect)
    
    ax.text(8, 9.2, 'AI-Enhanced SDN Control Plane', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # SDN Controller components
    controller_box = FancyBboxPatch((2, 8.2), 3, 1.4, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='white', 
                                   edgecolor='green', linewidth=1)
    ax.add_patch(controller_box)
    ax.text(3.5, 8.9, 'SDN Controller\n• Flow Management\n• Route Optimization\n• Dynamic Adaptation', 
            ha='center', va='center', fontsize=9)
    
    # AI Decision Engine
    ai_box = FancyBboxPatch((6, 8.2), 4, 1.4, 
                           boxstyle="round,pad=0.05", 
                           facecolor='white', 
                           edgecolor='green', linewidth=1)
    ax.add_patch(ai_box)
    ax.text(8, 8.9, 'AI Decision Engine\n• Multi-Domain Routing\n• Real-time Adaptation\n• QoS Optimization\n• Emergency Response', 
            ha='center', va='center', fontsize=9)
    
    # Network APIs
    api_box = FancyBboxPatch((11, 8.2), 3, 1.4, 
                            boxstyle="round,pad=0.05", 
                            facecolor='white', 
                            edgecolor='green', linewidth=1)
    ax.add_patch(api_box)
    ax.text(12.5, 8.9, 'Network APIs\n• OpenFlow Protocol\n• Domain Interfaces\n• Security APIs', 
            ha='center', va='center', fontsize=9)
    
    # Layer 3: GAN-Based Trust Evaluation Layer (Middle)
    gan_rect = FancyBboxPatch((1, 5.8), 14, 1.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['gan'], 
                             edgecolor='darkorange', linewidth=2)
    ax.add_patch(gan_rect)
    
    ax.text(8, 7, 'GAN-Based Trust Evaluation & Security Layer', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # GAN Generator
    gen_box = FancyBboxPatch((1.5, 6), 3, 1.4, 
                            boxstyle="round,pad=0.05", 
                            facecolor='white', 
                            edgecolor='orange', linewidth=1)
    ax.add_patch(gen_box)
    ax.text(3, 6.7, 'GAN Generator\n• Synthetic Data Gen\n• Attack Simulation\n• Traffic Prediction', 
            ha='center', va='center', fontsize=9)
    
    # Trust Evaluator
    trust_box = FancyBboxPatch((5, 6), 3, 1.4, 
                              boxstyle="round,pad=0.05", 
                              facecolor=colors['trust'], 
                              edgecolor='purple', linewidth=1)
    ax.add_patch(trust_box)
    ax.text(6.5, 6.7, 'Trust Evaluator\n• Direct Trust\n• Indirect Trust\n• Energy Trust\n• Composite Score', 
            ha='center', va='center', fontsize=9)
    
    # GAN Discriminator
    disc_box = FancyBboxPatch((8.5, 6), 3, 1.4, 
                             boxstyle="round,pad=0.05", 
                             facecolor='white', 
                             edgecolor='orange', linewidth=1)
    ax.add_patch(disc_box)
    ax.text(10, 6.7, 'GAN Discriminator\n• Anomaly Detection\n• Malicious Node ID\n• Attack Classification', 
            ha='center', va='center', fontsize=9)
    
    # Security Engine
    sec_box = FancyBboxPatch((12, 6), 2.5, 1.4, 
                            boxstyle="round,pad=0.05", 
                            facecolor=colors['security'], 
                            edgecolor='purple', linewidth=1)
    ax.add_patch(sec_box)
    ax.text(13.25, 6.7, 'Security Engine\n• Threat Analysis\n• Defense Actions\n• Incident Response', 
            ha='center', va='center', fontsize=9)
    
    # Layer 4: Multi-Domain IoT Networks (Bottom)
    
    # Healthcare Domain
    health_rect = FancyBboxPatch((0.5, 2), 4.5, 3.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['healthcare'], 
                                edgecolor='darkred', linewidth=2)
    ax.add_patch(health_rect)
    
    ax.text(2.75, 5.2, 'Healthcare IoT Domain', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Healthcare devices
    devices_health = [
        (1.5, 4.5, 'Patient\nMonitors'),
        (2.5, 4.5, 'Medical\nSensors'),
        (3.5, 4.5, 'Smart\nActuators'),
        (4.2, 4.5, 'IoT\nGateways')
    ]
    
    for x, y, label in devices_health:
        device = Circle((x, y), 0.25, facecolor='white', edgecolor='red', linewidth=1)
        ax.add_patch(device)
        ax.text(x, y-0.5, label, ha='center', va='center', fontsize=8)
    
    # Healthcare characteristics
    ax.text(2.75, 3.5, '• High Trust Requirements (0.75 baseline)\n• Low Latency (5ms avg)\n• Privacy Critical (HIPAA)\n• Real-time Monitoring', 
            ha='center', va='center', fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Healthcare attacks
    ax.text(2.75, 2.5, 'Threats: Data Breach, Privacy Violations,\nDevice Tampering, DoS Attacks', 
            ha='center', va='center', fontsize=8, style='italic', color='darkred')
    
    # Transportation Domain
    transport_rect = FancyBboxPatch((5.75, 2), 4.5, 3.5, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor=colors['transport'], 
                                   edgecolor='darkgreen', linewidth=2)
    ax.add_patch(transport_rect)
    
    ax.text(8, 5.2, 'Transportation IoT Domain', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Transportation devices
    devices_transport = [
        (6.5, 4.5, 'V2V\nComm'),
        (7.5, 4.5, 'Traffic\nSensors'),
        (8.5, 4.5, 'Smart\nSignals'),
        (9.5, 4.5, 'RSUs')
    ]
    
    for x, y, label in devices_transport:
        device = Rectangle((x-0.25, y-0.25), 0.5, 0.5, facecolor='white', edgecolor='green', linewidth=1)
        ax.add_patch(device)
        ax.text(x, y-0.5, label, ha='center', va='center', fontsize=8)
    
    # Transportation characteristics  
    ax.text(8, 3.5, '• Moderate Trust (0.65 baseline)\n• Real-time Safety (15ms avg)\n• High Mobility\n• Weather Resilience', 
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Transportation attacks
    ax.text(8, 2.5, 'Threats: GPS Spoofing, V2X Jamming,\nTraffic Manipulation, Routing Attacks', 
            ha='center', va='center', fontsize=8, style='italic', color='darkgreen')
    
    # Underwater Domain
    underwater_rect = FancyBboxPatch((11, 2), 4.5, 3.5, 
                                    boxstyle="round,pad=0.1", 
                                    facecolor=colors['underwater'], 
                                    edgecolor='darkblue', linewidth=2)
    ax.add_patch(underwater_rect)
    
    ax.text(13.25, 5.2, 'Underwater IoT Domain', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Underwater devices
    devices_underwater = [
        (12, 4.5, 'Acoustic\nSensors'),
        (13, 4.5, 'Marine\nMonitors'),
        (14, 4.5, 'Depth\nSensors'),
        (14.8, 4.5, 'Buoy\nGateways')
    ]
    
    for x, y, label in devices_underwater:
        device = patches.RegularPolygon((x, y), 6, radius=0.25, facecolor='white', edgecolor='blue', linewidth=1)
        ax.add_patch(device)
        ax.text(x, y-0.5, label, ha='center', va='center', fontsize=8)
    
    # Underwater characteristics
    ax.text(13.25, 3.5, '• Low Trust (0.55 baseline)\n• High Latency (100ms avg)\n• Energy Constrained\n• Harsh Environment', 
            ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Underwater attacks
    ax.text(13.25, 2.5, 'Threats: Signal Jamming, Node Capture,\nEnergy Drain, Malicious Nodes', 
            ha='center', va='center', fontsize=8, style='italic', color='darkblue')
    
    # Data Flow Arrows
    
    # Upward data flow (devices to control plane)
    arrow_props = dict(arrowstyle='->', lw=2, color=colors['arrow'])
    
    # Healthcare to SDN
    ax.annotate('', xy=(3.5, 8), xytext=(2.75, 5.5), arrowprops=arrow_props)
    ax.text(2.5, 6.8, 'Patient Data\nTrust Metrics', ha='center', va='center', fontsize=8, 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    # Transportation to SDN  
    ax.annotate('', xy=(8, 8), xytext=(8, 5.5), arrowprops=arrow_props)
    ax.text(8.8, 6.8, 'Traffic Data\nV2X Messages', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    # Underwater to SDN
    ax.annotate('', xy=(12.5, 8), xytext=(13.25, 5.5), arrowprops=arrow_props)
    ax.text(13.8, 6.8, 'Sensor Data\nAcoustic Signals', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    # SDN to GAN layer
    ax.annotate('', xy=(6.5, 7.6), xytext=(8, 8.2), arrowprops=arrow_props)
    ax.text(7.2, 7.9, 'Network\nTelemetry', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    # GAN to Cloud
    ax.annotate('', xy=(8, 10), xytext=(8, 7.6), arrowprops=arrow_props)
    ax.text(8.8, 8.8, 'Trust Scores\nAnomaly Reports', ha='center', va='center', fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    # Downward control flow (control plane to devices)
    control_arrow_props = dict(arrowstyle='->', lw=2, color='purple', linestyle='--')
    
    # Cloud to domains (control signals)
    ax.annotate('', xy=(2.75, 5.5), xytext=(6, 10), arrowprops=control_arrow_props)
    ax.annotate('', xy=(8, 5.5), xytext=(8, 10), arrowprops=control_arrow_props)  
    ax.annotate('', xy=(13.25, 5.5), xytext=(10, 10), arrowprops=control_arrow_props)
    
    ax.text(0.8, 7.5, 'Control\nSignals', ha='center', va='center', fontsize=8, color='purple',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
    
    # Cross-domain communication arrows
    cross_arrow_props = dict(arrowstyle='<->', lw=1.5, color='gray', linestyle=':')
    
    # Healthcare <-> Transportation
    ax.annotate('', xy=(5.5, 3.5), xytext=(5, 3.5), arrowprops=cross_arrow_props)
    
    # Transportation <-> Underwater  
    ax.annotate('', xy=(11, 3.5), xytext=(10.5, 3.5), arrowprops=cross_arrow_props)
    
    ax.text(8, 1.5, 'Cross-Domain Trust Sharing & Coordination', 
            ha='center', va='center', fontsize=10, style='italic', color='gray')
    
    # Legend
    legend_box = FancyBboxPatch((0.5, 0.2), 15, 1, 
                               boxstyle="round,pad=0.05", 
                               facecolor='lightgray', 
                               edgecolor='black', linewidth=1)
    ax.add_patch(legend_box)
    
    ax.text(8, 0.9, 'Key Features & Innovations', ha='center', va='center', fontsize=12, fontweight='bold')
    
    legend_text = """
    • GAN-Enhanced Trust Evaluation: Synthetic attack generation and real-time anomaly detection across domains
    • Multi-Domain Routing: Unified framework supporting healthcare, transportation, and underwater IoT networks  
    • Dynamic SDN Control: AI-driven routing adaptation based on trust scores, QoS requirements, and security threats
    • Cross-Domain Intelligence: Trust sharing and coordinated defense across heterogeneous IoT environments
    • Energy-Aware Optimization: Trust-energy trade-off optimization for resource-constrained underwater deployments
    """
    
    ax.text(8, 0.5, legend_text, ha='center', va='center', fontsize=10)
    
    # Technical specifications box
    spec_box = FancyBboxPatch((12, 0.8), 3.8, 0.8, 
                             boxstyle="round,pad=0.05", 
                             facecolor='lightyellow', 
                             edgecolor='orange', linewidth=1)
    ax.add_patch(spec_box)
    
    specs_text = """Trust Baselines:
Healthcare: 0.75 | Transport: 0.65 | Underwater: 0.55
Attack Types: DDoS, Malicious Node, Energy Drain, Routing
Performance: 99.6% AUC | Privacy: ε-DP (ε=1.0)"""
    
    ax.text(14, 1.2, specs_text, ha='center', va='center', fontsize=9,
            bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.9))
    
    # Save the architecture diagram
    plt.tight_layout()
    plt.savefig('SecureRouteX_Architecture_Diagram.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('SecureRouteX_Architecture_Diagram.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("🏗️ SECUREROUTEX ARCHITECTURE DIAGRAM GENERATED!")
    print("="*60)
    print("📁 Files created:")
    print("   • SecureRouteX_Architecture_Diagram.png (High-res for papers)")
    print("   • SecureRouteX_Architecture_Diagram.pdf (Vector format)")
    print()
    print("🎯 Architecture Components:")
    print("   ✅ Cloud Infrastructure & Management Layer")
    print("   ✅ AI-Enhanced SDN Control Plane")
    print("   ✅ GAN-Based Trust Evaluation & Security Layer")
    print("   ✅ Multi-Domain IoT Networks (Healthcare/Transport/Underwater)")
    print("   ✅ Data Flow & Control Flow Visualization")
    print("   ✅ Cross-Domain Communication Paths")
    print("   ✅ Technical Specifications & Innovations")
    print()
    print("📊 Publication Ready:")
    print("   • High-resolution PNG for digital papers")
    print("   • Vector PDF for print publications")
    print("   • Professional color scheme and typography")
    print("   • Comprehensive technical details")
    
    return fig

# Execute the architecture generation
if __name__ == "__main__":
    fig = create_secureroutex_architecture()
    plt.show()