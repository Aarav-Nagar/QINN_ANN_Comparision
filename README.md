# Quantum-Inspired Neural Networks vs Traditional ANNs for Stock Price Prediction

**Author:** Aarav Nagar
**School:** FCS Innovation Academy
**Date:** November 2025

## 🎯 Project Summary
Comparison of Quantum-Inspired Neural Networks (QINN) vs Advanced Artificial Neural Networks (ANN) for multi-horizon stock price prediction across 15 stocks.

## 📊 Key Results
- **240 experiments completed** (15 stocks × 4 horizons × 2 models × 2 training periods)
- **QINN Accuracy:** 52.3% (avg directional prediction)
- **ANN Accuracy:** 51.1% (avg directional prediction)
- **Statistical Significance:** p < 0.05 (paired t-test)
- **Trade-off:** QINN is 6.3× slower but 1.2% more accurate

## 🔬 Methodology
- **Dataset:** 10 years daily data (2015-2025)
- **Stocks:** NVDA, TSLA, MS, C, BRK-B, TEAM, NKTR, PFE, UNH, CVX, XOM, TRIP, AVAV, LULU, BAC
- **Features:** 44 features (price data + 39 macro indicators)
- **Horizons:** 1-day, 15-day, 21-day, 63-day predictions
- **Models:**
  - ANN: Ensemble (Dense + CNN + LSTM) with Attention (2.07M params)
  - QINN: 3 quantum circuits (8 qubits, 3 layers) (56K params)

## 🚀 Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run single experiment
python main_experiment.py

# View results
python analyze_results.py
```

## 📁 Project Structure
- `models/` - Neural network architectures
- `utils/` - Data processing and configuration
- `results/` - Experiment outputs (JSON format)
- `preprocessing/` - Data preprocessing scripts

## 🛠️ Technologies Used
- Python 3.10
- PyTorch 2.0 (Deep Learning)
- PennyLane 0.30 (Quantum ML)
- Pandas, NumPy, Scikit-learn
- yfinance (Stock data)

## 📈 Results Summary
| Metric | ANN | QINN | Winner |
|--------|-----|------|--------|
| Accuracy | 51.1% | 52.3% | QINN (+1.2%) |
| RMSE | 0.124 | 0.135 | ANN (lower) |
| Training Time | 53s | 335s | ANN (6.3× faster) |
| Parameters | 2.07M | 56K | QINN (37× fewer) |

## 🔑 Key Findings
1. **QINN shows statistically significant improvement** in directional accuracy
2. **Trade-off identified:** Accuracy vs computational efficiency
3. **QINN excels at direction**, ANN better at exact prices
4. Both models struggle with longer prediction horizons (>21 days)

## 🎓 Research Contribution
First comprehensive comparison of QINN vs ensemble ANN architectures for multi-horizon financial prediction with rigorous statistical validation.

## 📧 Contact
aarav.nagar22@gmail.com
