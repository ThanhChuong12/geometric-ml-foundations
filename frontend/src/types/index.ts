export interface PredictionResult {
  digit: number;
  confidence: number;
}

export interface ApiResponse {
  baseline: PredictionResult;
  augmentation: PredictionResult;
  averaging: PredictionResult;
}
