import { useId } from "react";

/**
 * A document-unique SVG id. Fixed ids collide when the router keeps earlier screens mounted:
 * `url(#id)` then resolves to a hidden copy and the fill disappears.
 */
export function useSvgId(prefix: string) {
  return `${prefix}-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;
}
