// PointNet API service — Part 2
// Dùng cùng axios instance với Part 1 (api.ts)
import api from './api';
import { PointNetApiResponse, SampleCloudResponse } from '@/types/pointnet';

export interface PerturbationParams {
  rotation_x: number;
  rotation_y: number;
  rotation_z: number;
  noise_level: number;
  drop_ratio: number;
}

/**
 * Phân loại point cloud 3D với cả PointNet Basic và Full.
 */
export const classifyPointCloud = async (
  points: number[][],
  numPoints: number = 1024,
  perturbation?: Partial<PerturbationParams>
): Promise<PointNetApiResponse> => {
  const response = await api.post<PointNetApiResponse>('/part2/classify', {
    points,
    num_points: numPoints,
    rotation_x: perturbation?.rotation_x ?? 0,
    rotation_y: perturbation?.rotation_y ?? 0,
    rotation_z: perturbation?.rotation_z ?? 0,
    noise_level: perturbation?.noise_level ?? 0,
    drop_ratio:  perturbation?.drop_ratio  ?? 0,
  });
  return response.data;
};

/**
 * Fetch a sample point cloud by class name.
 */
export const getSampleCloud = async (className: string): Promise<SampleCloudResponse> => {
  const response = await api.get<SampleCloudResponse>(`/part2/sample/${className}`);
  return response.data;
};
