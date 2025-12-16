# CAMADRL: Context-Aware Multi-Agent Deep Reinforcement Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

## 🎯 Overview

**CAMADRL** is a cutting-edge deep reinforcement learning framework for intelligent electric vehicle (EV) charging coordination. The system enables autonomous agents to coordinate resource allocation across large-scale EV networks while dynamically adapting to real-time environmental conditions.

### Key Achievements
- 🏆 **92% coordination success rate**
- ⚡ **15% energy efficiency improvement**
- 💰 **10% operational cost reduction**
- 🔋 **20% grid strain decrease**
- 🚀 **2.3× faster convergence**

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone git@github.com:muddsairsharif/CAMADRL.git
cd CAMADRL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Demo

```bash
# Quick demo (5 minutes)
python scripts/demo.py

# Full training
python scripts/train.py --episodes 150
```

## 📁 Repository Structure

```
CAMADRL/
├── src/              # Source code
│   ├── models/       # Neural network models
│   ├── environment/  # Simulation environment
│   ├── training/     # Training utilities
│   └── utils/        # Helper functions
├── scripts/          # Executable scripts
├── tests/            # Unit tests
├── docs/             # Documentation
├── config/           # Configuration files
└── data/             # Datasets
```

## 🏗️ Architecture

- **Graph Neural Networks** for infrastructure modeling
- **Multi-Head Attention** for context processing
- **Multi-Stakeholder Q-Networks** for optimization
- **Hierarchical Coordination** using PSO/GA

## 📊 Results

| Metric | CAMADRL | Baseline |
|--------|---------|----------|
| Coordination Success | 92% | 78% |
| Energy Efficiency | +15% | +8% |
| Cost Reduction | 10% | 5% |
| Convergence Speed | 15 eps | 35 eps |

## 📖 Documentation

- [Architecture Guide](docs/architecture.md)
- [Weekly Development Guide](docs/weekly_guide.md)
- [API Reference](docs/api_reference.md)
- [Troubleshooting](docs/troubleshooting.md)

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 Citation

```bibtex
@article{sharif2025camadrl,
  title={Context-Aware Multi-Agent Coordination Framework for Intelligent Electric Vehicle Charging Optimization},
  author={Sharif, Muddsair and Seker, Huseyin and Javed, Yasir},
  journal={IEEE Access},
  year={2025}
}
```

## 📧 Contact

**Muddsair Sharif**  
Stuttgart University of Applied Sciences  
📧 muddsair.sharif@hft-stuttgart.de  
🔗 [GitHub](https://github.com/muddsairsharif)

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

---

⭐ **Star this repository if you find it useful!**