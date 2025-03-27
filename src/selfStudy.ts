export interface DataPoint {
  features: number[];
  label: string;
}
const beta = 3.0;
const radiusCrit = 0.01;

export class SelfStudy {
  // Коэффициент скорости обучения
  private readonly learningRate: number;
  // Кол-во нейронов
  private readonly countOfNeurons: number;
  // Веса нейронов
  private weights: number[][];
  // Потенциалы нейронов
  private potents: Array<number> = [];

  constructor(
    countOfNeurons: number,
    inputDimension: number,
    learningRate: number
  ) {
    this.countOfNeurons = countOfNeurons;
    //this.inputDimension = inputDimension;
    this.learningRate = learningRate;

    // Инициализируем веса случайно
    this.weights = Array.from({ length: countOfNeurons }, (_, r) =>
      Array.from({ length: inputDimension }, (_, c) => Math.random()/20)
    );
    // Инициализируем потенциалы
    this.potents = Array.from(
      { length: countOfNeurons },
      (_, r) => 1 / countOfNeurons
    );
  }

  // Находит индекс победителя
  public findBestMatchingUnit(input: number[]): number {
    let minDistance = Number.POSITIVE_INFINITY;
    // Индекс самого близкого нейрона
    let bestNeuronIndex = 0;

    for (
      let neuronIndex = 0;
      neuronIndex < this.countOfNeurons;
      neuronIndex++
    ) {
      if (this.potents[neuronIndex] < 0) {
        continue;
      }
      const weights = this.weights[neuronIndex];
      const distance = this.calculateEuclideanDistance(weights, input);
      if (distance < minDistance) {
        minDistance = distance;
        bestNeuronIndex = neuronIndex;
      }
    }
    return bestNeuronIndex;
  }
  // Находит индекс победителя
  public findBestMatchingUnitArray(input: number[]): number[] {
    let radiuses: number[] = [];

    for (
      let neuronIndex = 0;
      neuronIndex < this.countOfNeurons;
      neuronIndex++
    ) {
      if (this.potents[neuronIndex] < 0) {
        continue;
      }
      const weights = this.weights[neuronIndex];
      const distance = this.calculateEuclideanDistance(weights, input);
      radiuses.push(distance);
    }
    return radiuses;
  }
  // Обучает сеть
  public train(data: DataPoint[], epochs: number, minTreshold: number) {
    let sumDelta: number;
    for (let t = 0; t < epochs; t++) {
      sumDelta = 0;
      for (let i = 0; i < data.length; ++i) {
        // Находим индекс победителя
        const indexOfWinner: number = this.findBestMatchingUnit(
          data[i].features
        );

        // Обновим веса победителя
        this.weights[indexOfWinner].forEach(
          (weight: number, indexOfInputData: number) => {
            const delta: number = data[i].features[indexOfInputData] - weight;
            sumDelta += Math.abs(delta);
            return weight + this.learningRate * delta;
          }
        );

        for (let index: number = 0; index < this.countOfNeurons; ++index) {
          if (index === indexOfWinner) {
            this.potents[index] -= 0.75;
          } else {
            this.potents[index] += 1 / this.countOfNeurons;
          }
        }
      }
      if (sumDelta < minTreshold) {
        break;
      }
    }
    this.potents = this.potents = Array.from(
      { length: this.countOfNeurons },
      (_, r) => 1 / this.countOfNeurons
    );
  }

  public trainArray(data: DataPoint[], epochs: number, minTreshold: number) {
    let sumDelta: number;
    for (let t = 0; t < epochs; t++) {
      sumDelta = 0;
      for (let i = 0; i < data.length; ++i) {
        // Находим индекс победителя
        const radiuses: number[] = this.findBestMatchingUnitArray(
          data[i].features
        );
        this.weights.forEach((weight, idx) => {
          weight.forEach((weight: number, indexOfInputData: number) => {
            const delta: number = data[i].features[indexOfInputData] - weight;
            sumDelta += Math.abs(delta);
            const v: number =
              1 - 1 / (1 + Math.exp(-beta * (radiuses[idx] - radiusCrit)));

            return weight + this.learningRate * delta * v;
          });
        });
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
    return Math.sqrt(a.reduce((acc, val, idx) => acc + (val - b[idx]) ** 2, 0));
  }
}
