// PointNet TypeScript types — Part 2
// Không ảnh hưởng đến types/index.ts của Part 1

export interface Top3Prediction {
  label: string;
  class_id: number;
  confidence: number;
}

export interface PointNetModelResult {
  label: string;
  class_id: number;
  confidence: number;
  top3: Top3Prediction[];
  critical_points: number[][];
  num_critical: number;
}

export interface PointNetApiResponse {
  basic_model: PointNetModelResult;
  full_model: PointNetModelResult;
  point_cloud: number[][];
  num_points_used: number;
  processing_time_ms: number;
}

export interface SampleCloudResponse {
  class_name: string;
  points: number[][];
  num_points: number;
}

// Demo classes available in the dropdown
export const DEMO_CLASSES = ['airplane', 'chair', 'car', 'lamp', 'table'] as const;
export type DemoClass = typeof DEMO_CLASSES[number];

export const CLASS_ICONS: Record<DemoClass, string> = {
  airplane: 'Airplane',
  chair:    'Chair',
  car:      'Car',
  lamp:     'Lamp',
  table:    'Table',
};
