#!/bin/bash
# setup.sh — One-command environment setup for smartdaas-hiv-validation
# Usage: bash setup.sh

echo "Setting up smartdaas-hiv-validation environment..."

# Create required directories
mkdir -p data results figures notebooks app

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete. To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the full pipeline:"
echo "  python src/01_data_preprocessing.py"
echo "  python src/02_model_training_cv.py"
echo "  python src/03_temporal_validation.py"
echo "  python src/04_shap_explainability.py"
echo "  python src/05_decision_curve_analysis.py"
echo "  python src/06_subgroup_fairness.py"
echo "  python src/07_economic_modelling.py"
echo "  python src/08_figures.py"
echo ""
echo "To launch the interactive demo:"
echo "  streamlit run app/demo.py"
