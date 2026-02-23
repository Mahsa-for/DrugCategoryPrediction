from interactive_predictor import InteractiveDrugPredictor
p = InteractiveDrugPredictor()
print('genes x regions:', p.gene_expression.shape)
print('scaler loaded:', p.feature_scaler is not None)
feat, inp, avail = p.extract_features(['F2','EGFR','INVALIDGENE'])
print('feature shape:', None if feat is None else feat.shape)
print('available genes sample:', avail[:10])
