"""
Flask API Backend for Drug Category Prediction
Serves predictions via REST API for the web frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from interactive_predictor import InteractiveDrugPredictor
from src.metrics.brain_evidence import BrainEvidenceMetrics

app = Flask(__name__)
CORS(app)

# Initialize predictor and brain evidence metrics
predictor = None
brain_metrics = BrainEvidenceMetrics(tau_strength=0.3, tau_ratio=0.6)

@app.before_request
def initialize():
    """Initialize predictor on first request"""
    global predictor
    if predictor is None:
        print("Initializing predictor...")
        predictor = InteractiveDrugPredictor()
        if not predictor.classifier_model:
            return {"error": "Models not loaded"}, 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if predictor and predictor.classifier_model:
        return {
            'status': 'ok',
            'message': 'Drug prediction service is running',
            'models_loaded': True,
            'num_genes': predictor.gene_expression.shape[0],
            'num_categories': len(predictor.label_encoder.classes_),
            'categories': predictor.label_encoder.classes_.tolist()
        }
    return {"status": "error", "message": "Models not loaded"}, 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict drug category from genes"""
    try:
        data = request.json
        genes = data.get('genes', [])
        top_k = data.get('top_k', 5)
        
        if not genes:
            return {"error": "No genes provided"}, 400
        
        # Clean genes
        genes = [g.strip().upper() for g in genes if isinstance(g, str) and g.strip()]
        
        if not genes:
            return {"error": "No valid genes provided"}, 400
        
        # Get prediction
        result = predictor.predict(genes, top_k=top_k)
        
        return jsonify(result)
    
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """Batch predict drug categories"""
    try:
        data = request.json
        drugs = data.get('drugs', {})
        top_k = data.get('top_k', 3)
        
        if not drugs:
            return {"error": "No drugs provided"}, 400
        
        results = {}
        for drug_name, genes in drugs.items():
            genes = [g.strip().upper() for g in genes if isinstance(g, str) and g.strip()]
            if genes:
                results[drug_name] = predictor.predict(genes, top_k=top_k)
        
        return jsonify(results)
    
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/genes', methods=['GET'])
def get_genes():
    """Get list of available genes"""
    try:
        if predictor and predictor.gene_expression is not None:
            genes = predictor.gene_expression.index.tolist()
            return jsonify({
                'genes': genes,
                'total': len(genes)
            })
        return {"error": "Gene data not loaded"}, 500
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get ATC categories"""
    try:
        categories = []
        for code, name in predictor.atc_descriptions.items():
            categories.append({'code': code, 'name': name})
        return jsonify(categories)
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Get example drug predictions"""
    examples = {
        'Antipsychotic': {
            'description': 'Dopamine D2 antagonist - blocks dopamine signaling',
            'genes': ['DRD2', 'DRD3', 'HTR2A']
        },
        'Beta-blocker': {
            'description': 'Cardiovascular agent - blocks adrenergic receptors',
            'genes': ['ADRB1', 'ADRB2']
        },
        'Cancer Drug': {
            'description': 'Tyrosine kinase inhibitor - blocks growth signals',
            'genes': ['EGFR', 'ERBB2', 'TP53']
        },
        'Opioid': {
            'description': 'Pain management - acts on opioid receptors',
            'genes': ['OPRM1', 'OPRD1', 'OPRK1']
        },
        'Antibiotic': {
            'description': 'Bacterial topoisomerase inhibitor',
            'genes': ['GYRA', 'GYRB']
        }
    }
    
    # Get predictions for each example
    results = {}
    for name, example in examples.items():
        result = predictor.predict(example['genes'], top_k=3)
        results[name] = {
            'description': example['description'],
            'genes': example['genes'],
            'prediction': result
        }
    
    return jsonify(results)

if __name__ == '__main__':
    print("Starting Drug Prediction API Server...")
    print("Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/predict - Single prediction")
    print("  POST /api/predict-batch - Batch prediction")
    print("  GET  /api/genes - List available genes")
    print("  GET  /api/categories - List ATC categories")
    print("  GET  /api/examples - Get example predictions")
    print("\nServer running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
