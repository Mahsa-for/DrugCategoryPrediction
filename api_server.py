"""
Flask API Backend for Drug Category Prediction
Serves predictions and explanations for the web UI
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from interactive_predictor import InteractiveDrugPredictor
from src.agents.evidence_agent import EvidenceAgent, EvidenceInputs

app = Flask(__name__)
CORS(app)

# Initialize predictor and agent
predictor = None
evidence_agent = EvidenceAgent()
RESULTS_DIR = Path(__file__).parent / "results"

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
    """Health check endpoint for UI"""
    if predictor and predictor.classifier_model:
        return {
            'status': 'ok',
            'message': 'Drug prediction service is running',
            'models_loaded': True,
            'num_genes': predictor.gene_expression.shape[0],
            'num_categories': len(predictor.label_encoder.classes_),
            'categories': predictor.label_encoder.classes_.tolist(),
        }
    return {"status": "error", "message": "Models not loaded"}, 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict drug category from genes"""
    try:
        data = request.json
        genes = data.get('genes', [])
        top_k = data.get('top_k', 3)
        if not genes:
            return {"error": "No genes provided"}, 400
        genes = [g.strip().upper() for g in genes if isinstance(g, str) and g.strip()]
        if not genes:
            return {"error": "No valid genes provided"}, 400
        result = predictor.predict(genes, top_k=top_k)
        # Always resolve category names for all predictions
        def resolve_category_name(code):
            return predictor.atc_descriptions.get(code, 'Unknown Category')

        # Patch primary_prediction and top_predictions with correct names
        if 'primary_prediction' in result and result['primary_prediction']:
            code = result['primary_prediction'].get('category_code')
            if code:
                result['primary_prediction']['category_name'] = resolve_category_name(code)
        if 'top_predictions' in result:
            for pred in result['top_predictions']:
                code = pred.get('category_code')
                if code:
                    pred['category_name'] = resolve_category_name(code)

        # Add metric explanations from brain_evidence.py docstrings
        metric_explanations = {
            'BES': {
                'name': 'Brain Evidence Strength',
                'description': 'Measures the average expression level across brain cell cluster types for target genes. Higher BES indicates stronger molecular evidence in the brain. Range: 0-1.'
            },
            'BSR': {
                'name': 'Brain Specificity Ratio',
                'description': 'Measures the relative expression strength in brain cell cluster types versus body tissues. Higher BSR indicates that drug effects are preferentially observable in brain. BSR > 1: brain-enriched; BSR < 1: body-enriched.'
            },
            'BSF': {
                'name': 'Brain Sufficiency Flag',
                'description': 'Boolean indicator of whether brain evidence is sufficient to support a brain-based interpretation of the predicted drug category. True if BES and BSR pass thresholds.'
            },
            'CNS Score': {
                'name': 'CNS Score',
                'description': 'Combined score from BES and BSR, indicating likelihood of CNS activity.'
            },
            'Gene Coverage': {
                'name': 'Gene Coverage',
                'description': 'Fraction of input genes found in the expression database.'
            }
        }
        result['metric_explanations'] = metric_explanations

        # Convert all numpy types to Python types for JSON serialization
        def convert(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            else:
                return obj
        result_clean = convert(result)
        return jsonify(result_clean)
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/genes', methods=['GET'])
def get_genes():
    """Get list of available genes for UI autocomplete"""
    try:
        if predictor and predictor.gene_expression is not None:
            genes = predictor.gene_expression.index.tolist()
            return jsonify({'genes': genes, 'total': len(genes)})
        return {"error": "Gene data not loaded"}, 500
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get ATC categories for UI display"""
    try:
        categories = []
        for code, name in predictor.atc_descriptions.items():
            categories.append({'code': code, 'name': name})
        return jsonify(categories)
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/explain-prediction', methods=['POST'])
def explain_prediction():
    """Get evidence + reasoning for a prediction (calls EvidenceAgent, which uses TxAgent)"""
    try:
        data = request.json
        drug_name = data.get('drug_name', 'Unknown')
        predicted_category = data.get('predicted_category', '')
        target_genes = data.get('target_genes', [])
        model_probs = data.get('model_probs', {})

        # Get CNS relevance and brain evidence from backend
        cns_info = {}
        brain_evidence = {}
        gene_coverage = 0.0
        brain_top_clusters = None
        extracted_model_probs = {}

        if predictor and predictor.gene_expression is not None:
            cns_info = predictor.get_cns_relevance_for_genes(target_genes)
            pred_result = predictor.predict(target_genes, top_k=1)
            if 'brain_evidence' in pred_result:
                be = pred_result['brain_evidence']
                brain_evidence = {
                    'bes': float(be.get('bes', 0.0)),
                    'bsr': float(be.get('bsr', 0.0)) if be.get('bsr') != 'inf' else float('inf')
                }
            if 'gene_coverage' in pred_result:
                gene_coverage = float(pred_result['gene_coverage'])
            if 'brain_top_clusters' in pred_result:
                brain_top_clusters = pred_result['brain_top_clusters']
            if 'model_probs' in pred_result:
                extracted_model_probs = pred_result['model_probs']

        final_model_probs = model_probs or extracted_model_probs or {}

        # Extract full features vector for agent
        features_vector = None
        if predictor and predictor.gene_expression is not None:
            features_vector, _, _ = predictor.extract_features(target_genes)
            if features_vector is not None:
                features_vector = features_vector.flatten().tolist()

        inp = EvidenceInputs(
            drug_name=drug_name,
            predicted_category=predicted_category,
            target_genes=target_genes,
            model_probs=final_model_probs,
            cns_score=float(cns_info.get('cns_probability', 0.0)),
            bes=brain_evidence.get('bes'),
            bsr=brain_evidence.get('bsr') if brain_evidence.get('bsr') != float('inf') else 999.0,
            gene_coverage=gene_coverage if gene_coverage > 0 else None,
            brain_top_clusters=brain_top_clusters,
            features=features_vector
        )

        explanation = evidence_agent.run(inp)
        explanation["evidence_inputs"] = {
            "cns_score": float(cns_info.get("cns_probability", 0.0)),
            "bes": brain_evidence.get("bes"),
            "bsr": brain_evidence.get("bsr"),
            "gene_coverage": gene_coverage if gene_coverage > 0 else None,
            "brain_top_clusters": brain_top_clusters,
        }

        # Convert all numpy types to Python types for JSON serialization
        def convert(obj):
            import numpy as np
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            else:
                return obj
        explanation_clean = convert(explanation)
        return jsonify(explanation_clean)
    except Exception as e:
        print(f"[ERROR] explain-prediction failed: {str(e)}")
        return {"error": str(e)}, 400

# Static file routes for web interface
@app.route('/')
def index():
    """Serve the main web interface"""
    return send_from_directory('web', 'index.html')

@app.route('/web/<path:filename>')
def web_static(filename):
    """Serve static files from web directory"""
    return send_from_directory('web', filename)

if __name__ == '__main__':
    print("Starting Drug Prediction API Server...")
    print("Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  POST /api/predict - Single prediction")
    print("  GET  /api/genes - List available genes")
    print("  GET  /api/categories - List ATC categories")
    print("  POST /api/explain-prediction - Explain prediction with evidence + reasoning")
    print("\nServer running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
