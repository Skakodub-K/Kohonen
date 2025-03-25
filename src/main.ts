import fs from 'fs';
import { DataPoint } from "./selfStudy";
import { SelfStudy } from "./selfStudy";
import { SelfOrganization } from "./selfOrganization";
import { features } from "process";

async function readJsonFile(filePath: string): Promise<any> {
  try {
    const data = await fs.promises.readFile(filePath, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    console.error('Ошибка при чтении файла:', error);
    throw error;
  }
}

// Набор данных
const dataPoints: DataPoint[] = [];

async function processData(): Promise<boolean> {
  let dataObj: any;
  try {
    // Читаем файл-дата
    const filePath = "data.json";
    dataObj = await readJsonFile(filePath);
    console.log("Объект из JSON файла прочитан");
  } catch (error) {
    console.error("Ошибка чтения данных:", error);
    return false;
  }
  // Сумма квадратов входных данных
  const squareSumm: number[] = [0, 0, 0, 0];

  for (let countryName in dataObj) {
    const features: number[] = [];
    for (let fetname in dataObj[countryName]) {
      features.push(dataObj[countryName][fetname]);
    }

    // Используем для вычисления суммы квадратов
    squareSumm.forEach((_, index) => {
      squareSumm[index] += features[index] * features[index];
    });

    dataPoints.push({ label: countryName, features });
  }

  // Нормализуем
  for (let i = 0; i < dataPoints.length; ++i) {
    dataPoints[i].features = dataPoints[i].features.map(
      (x: number, index: number) => x / Math.sqrt(squareSumm[index])
    );
  }

  console.log("Набор данных обработан");
  return true;
}

// Пример1 с самообучающейся сетью Кохонера
async function Example1(countOfNeurons:number) {
  // Размер входного вектора
  const inputDimension = 2;
  // Скорость обучения
  const learningRate = 0.4;
  // Количество эпох
  let epochs: number = 3000;
  // Минимальный порог, после которого прекращается обучение
  let delta: number = 0.1;

  const som = new SelfStudy(countOfNeurons, inputDimension, learningRate);

  const dataPointsForSS: Array<DataPoint> = dataPoints.map(
    (data: DataPoint) => {
      const newData: DataPoint = {
        label: data.label,
        features: [data.features[0], data.features[1]]
      };
      return newData;
    }
  );

  // Тренериуем
  som.train(dataPointsForSS, epochs, delta);

  // Теперь проверяем все данные
  const clusters: string[][] = Array.from({ length: countOfNeurons }, () => []);
  for (const data of dataPointsForSS) {
    clusters[som.findBestMatchingUnit(data.features)].push(data.label);
  }
  let i = 1;
  for (const cluster of clusters) {
    console.log("Кластер ", i++, " его содкржимое:");
    let countrys = "";
    cluster.forEach((country: string) => {
      countrys += " " + country;
    });
    console.log(countrys);
  }
}
// Пример2
async function Example2(radius:number) {
  // Размер входного вектора
  const inputDimension = 2;
  // Количество эпох
  let epochs: number = 3000;
  // Минимальный порог, после которого прекращается обучение
  let delta: number = 0.1;
  // Скорость обучения
  const learningRate = 0.4;

  const sorg = new SelfOrganization(radius, inputDimension, learningRate);

  const dataPointsForSO: Array<DataPoint> = dataPoints.map(
    (data: DataPoint) => {
      const newData: DataPoint = {
        label: data.label,
        features: [data.features[2], data.features[3]]
      };
      return newData;
    }
  );

  sorg.train(dataPointsForSO, epochs, delta);

  // Создаем карту кластеров
  const clusters: Map<number, string[]> = new Map();

  // Заполняем карту кластерами
  for (const data of dataPointsForSO) {
    // Получаем индекс лучшего соответствия
    const unitIndex: number = sorg.findBestMatchingUnit(data.features);
    // Если кластера еще нет, создаем его
    if (!clusters.has(unitIndex)) {
      clusters.set(unitIndex, []);
    }
    // Добавляем страну в соответствующий кластер
    clusters.get(unitIndex)?.push(data.label);
  }

  // Выводим содержимое каждого кластера
  for (const [clusterNumber, countries] of clusters.entries()) {
    console.log(`Кластер ${clusterNumber}:`);
    console.log(countries.join(', ')); // Объединяем страны в строку через запятую
  }
}
async function main() {
  const args = process.argv.slice(2);
  if (!await processData()) {
    console.log("FATAL ERROR");
    return;
  }

  if (args.includes("--example1")) {
    const example1Index = args.indexOf('--example1');
    const clusterCount:number = parseInt(args[example1Index + 1]) || 11; // По умолчанию 11 кластеров
    await Example1(clusterCount);
  }

  if (args.includes("--example2")) {
    const example2Index = args.indexOf('--example2');
    const radius:number = parseInt(args[example2Index + 1]) || 0.019; // По умолчанию радиус 
    await Example2(radius);
  }
}

main();
