
import { DataPoint } from "./selfStudy";

export class SelfOrganization {
  // Коэффициент скорости обучения
  private readonly learningRate: number;
  // Радиус кластера
  private readonly radius: number;
  // Веса нейронов
  private weights: number[][] = [];

  constructor(radius: number, inputDimension: number, learningRate: number) {
    this.radius = radius;
    //this.inputDimension = inputDimension;
    this.learningRate = learningRate;
  }

  // Находит индекс победителя
  public findBestMatchingUnit(input: number[]): number {
    let minDistance = Number.POSITIVE_INFINITY;
    // Индекс самого близкого нейрона
    let bestNeuronIndex = -1;

    for (let neuronIndex = 0; neuronIndex < this.weights.length; neuronIndex++) {
      const weights = this.weights[neuronIndex];
      const distance = this.calculateEuclideanDistance(weights, input);
      if (distance < minDistance && distance < this.radius) {
        minDistance = distance;
        bestNeuronIndex = neuronIndex;
      }
    }
    return bestNeuronIndex;
  }

  // Обучает сеть
  public train(data: DataPoint[], epochs: number, minTreshold: number) {
    let sumDelta: number;
    for (let t = 0; t < epochs; t++) {
      sumDelta = 0;
      for (let i = 0; i < data.length; ++i) {
        // Находим индекс победителя
        const indexOfWinner: number = this.findBestMatchingUnit(data[i].features);
        if (indexOfWinner === -1) {
          // Значит, добавляем новый нейрон
          this.weights.push([...data[i].features]);
          continue;
        }
        // Обновим веса победителя
        this.weights[indexOfWinner].forEach((weight: number, index: number) => {
          const delta: number = data[i].features[index] - weight;
          sumDelta += Math.abs(delta);
          return weight + this.learningRate * delta;
        })
      }
      if (sumDelta < minTreshold) {
        break;
      }
    }
  }

  public getWeights(): number[][] {
    return this.weights;
  }

  // Считает Евклидово расстояние между двумя векторами
  private calculateEuclideanDistance(a: number[], b: number[]) {
    return Math.sqrt(a.reduce((acc, val, idx) => acc + ((val - b[idx]) ** 2), 0));
  }
}