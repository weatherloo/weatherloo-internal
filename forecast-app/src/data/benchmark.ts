import summary from "./benchmark-summary.json";

export interface BenchmarkScore {
  rmse: number;
  n_samples?: number;
}

export interface ModelBenchmark {
  n_inits: number;
  t2m: BenchmarkScore | null;
  wind_speed: BenchmarkScore | null;
}

export const BENCHMARK_YEAR = summary.year;
export const BENCHMARK_STATION_ID = summary.station_id;
export const BENCHMARK_LEAD_HOURS = summary.lead_time_hours;

export const getBenchmark = (methodId?: string): ModelBenchmark | undefined =>
  methodId ? summary.methods[methodId as keyof typeof summary.methods] : undefined;
