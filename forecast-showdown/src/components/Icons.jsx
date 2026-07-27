/* Inline stroke icons — no icon dependency, so the bundle stays self-contained. */

const base = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.8,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
}

export const Sun = (p) => (
  <svg {...base} {...p}>
    <circle cx="12" cy="12" r="4.2" />
    <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" />
  </svg>
)

export const Cloud = (p) => (
  <svg {...base} {...p}>
    <path d="M17.5 18a4 4 0 0 0 .3-8 6 6 0 0 0-11.5 1.6A3.5 3.5 0 0 0 7 18Z" />
  </svg>
)

export const PartlyCloud = (p) => (
  <svg {...base} {...p}>
    <circle cx="8" cy="8" r="3" />
    <path d="M8 1.5v1.6M1.5 8h1.6M3.6 3.6l1.1 1.1M12.4 3.6l-1.1 1.1" />
    <path d="M17.5 19a3.5 3.5 0 0 0 .2-7 5.5 5.5 0 0 0-10.4 1.2A3.2 3.2 0 0 0 7.6 19Z" />
  </svg>
)

export const Moon = (p) => (
  <svg {...base} {...p}>
    <path d="M20 14.2A8.2 8.2 0 0 1 9.8 4a8.4 8.4 0 1 0 10.2 10.2Z" />
  </svg>
)

export const Pin = (p) => (
  <svg {...base} {...p}>
    <path d="M20 10.5c0 5.2-8 12-8 12s-8-6.8-8-12a8 8 0 1 1 16 0Z" />
    <circle cx="12" cy="10.5" r="2.8" />
  </svg>
)

export const Chevron = (p) => (
  <svg {...base} {...p}><path d="m6 9 6 6 6-6" /></svg>
)

export const Check = (p) => (
  <svg {...base} {...p} strokeWidth="2.6"><path d="m4 12.5 5.2 5L20 7" /></svg>
)

export const Trophy = (p) => (
  <svg {...base} {...p}>
    <path d="M7 4h10v5a5 5 0 0 1-10 0Z" />
    <path d="M17 5h3v2a3 3 0 0 1-3 3M7 5H4v2a3 3 0 0 0 3 3" />
    <path d="M12 14v3M9 20h6M10 17h4" />
  </svg>
)

export const Chart = (p) => (
  <svg {...base} {...p}>
    <path d="M4 19V5M4 19h16" />
    <path d="m7.5 14.5 3.5-4 3 2.6 4.5-6" />
  </svg>
)

export const Layers = (p) => (
  <svg {...base} {...p}>
    <path d="m12 3 8.5 4.5L12 12 3.5 7.5 12 3Z" />
    <path d="m3.5 12.2 8.5 4.5 8.5-4.5M3.5 16.7 12 21.2l8.5-4.5" />
  </svg>
)

export const Station = (p) => (
  <svg {...base} {...p}>
    <path d="M12 13v8M8.5 21h7" />
    <circle cx="12" cy="9" r="2.2" />
    <path d="M7.4 14.1a7 7 0 0 1 0-10.2M16.6 3.9a7 7 0 0 1 0 10.2" />
  </svg>
)

export const Info = (p) => (
  <svg {...base} {...p}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 11v5.5M12 7.7v.5" />
  </svg>
)

export const Warn = (p) => (
  <svg {...base} {...p}>
    <path d="M10.3 3.9 2.4 17.4A2 2 0 0 0 4.1 20.4h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z" />
    <path d="M12 9.4v4.2M12 17.1v.4" />
  </svg>
)

export const Moon2 = (p) => (
  <svg {...base} {...p}>
    <path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11Z" />
  </svg>
)

export const SunSmall = (p) => (
  <svg {...base} {...p}>
    <circle cx="12" cy="12" r="4.6" />
    <path d="M12 2.4v1.8M12 19.8v1.8M2.4 12h1.8M19.8 12h1.8M5.2 5.2l1.3 1.3M17.5 17.5l1.3 1.3M18.8 5.2l-1.3 1.3M6.5 17.5l-1.3 1.3" />
  </svg>
)

export const Wind = (p) => (
  <svg {...base} {...p}>
    <path d="M3 8h11a3 3 0 1 0-3-3M3 12h15a3 3 0 1 1-3 3M3 16h8" />
  </svg>
)

export const Ruler = (p) => (
  <svg {...base} {...p}>
    <path d="M3.5 15.5 15.5 3.5l5 5-12 12z" />
    <path d="M7 12l1.8 1.8M10 9l1.8 1.8M13 6l1.8 1.8" />
  </svg>
)

export const Clock = (p) => (
  <svg {...base} {...p}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5.2l3.2 2" />
  </svg>
)

export const Target = (p) => (
  <svg {...base} {...p}>
    <circle cx="12" cy="12" r="8.5" /><circle cx="12" cy="12" r="4.6" /><circle cx="12" cy="12" r="1" />
  </svg>
)

export const Grid = (p) => (
  <svg {...base} {...p}>
    <rect x="3.5" y="3.5" width="7" height="7" rx="1.6" />
    <rect x="13.5" y="3.5" width="7" height="7" rx="1.6" />
    <rect x="3.5" y="13.5" width="7" height="7" rx="1.6" />
    <rect x="13.5" y="13.5" width="7" height="7" rx="1.6" />
  </svg>
)

export const Bolt = (p) => (
  <svg {...base} {...p}><path d="M13.2 2.5 4.8 13.2h6l-1.2 8.3 8.6-10.8h-6.2z" /></svg>
)

/** Maps skyOf() output to an icon component. */
export const SKY = {
  sun: Sun,
  partly: PartlyCloud,
  cloud: Cloud,
  night: Moon,
  day: Sun,
}
