/** Station metadata. IDs match benchmarking-site/AGENTS.md exactly. */
export interface Station {
  id: string;
  name: string;
  place: string; // shown in the header
  lat: number;
  lon: number;
  timeZone: string; // IANA; display only, all data stays UTC
}

export const STATIONS: Station[] = [
  {
    id: "eric_d_soulis",
    name: "Eric D. Soulis Station",
    place: "Waterloo, Ontario",
    lat: 43.4723,
    lon: -80.5449,
    timeZone: "America/Toronto",
  },
];
export const DEFAULT_STATION_ID = "eric_d_soulis";
export const getStation = (id: string) => STATIONS.find((s) => s.id === id) ?? STATIONS[0];
