import api from './api';
import { ApiResponse } from '@/types';

export const predictDigit = async (imageData: string): Promise<ApiResponse> => {
  const response = await api.post<ApiResponse>('/part1/predict', {
    image: imageData,
  });
  return response.data;
};
