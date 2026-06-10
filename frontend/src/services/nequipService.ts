import api from './api';
import { MoleculeInput } from '@/types/molecule';
import { EnergyPredictionResponse } from '@/types/prediction';

/**
 * Predicts the quantum chemical molecular energy (U0) using the NequIP backend service.
 * 
 * @param molecule The object containing atomic numbers and positions.
 * @returns A promise resolving to the EnergyPredictionResponse.
 */
export async function predictEnergy(
  molecule: MoleculeInput
): Promise<EnergyPredictionResponse> {
  try {
    const response = await api.post<EnergyPredictionResponse>('/part3/predict-energy', {
      atomic_numbers: molecule.atomic_numbers,
      positions: molecule.positions,
    });
    return response.data;
  } catch (error: any) {
    // Extract backend HTTP error details if available
    if (error.response?.data?.detail) {
      const detail = error.response.data.detail;
      if (typeof detail === 'string') {
        throw new Error(detail);
      } else if (Array.isArray(detail)) {
        // Handle Pydantic list of validation errors
        throw new Error(detail.map((d: any) => d.msg || d.message || JSON.stringify(d)).join(', '));
      }
    }
    throw new Error(error.message || 'Không thể kết nối đến backend dự đoán năng lượng NequIP.');
  }
}
