import type { Config } from "tailwindcss";

// Adobe Spectrum-inspired light theme: restrained neutral grays, generous
// whitespace, a single Adobe red accent. Type uses Source Sans 3 (open
// stand-in for Adobe Clean) with Inter / system fallbacks.
const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        adobe: {
          red: "#EB1000",
          redDark: "#B30B00",
          redTint: "#FBE7E5",
        },
        spectrum: {
          bg: "#FFFFFF",
          surface: "#F5F5F5",
          surface2: "#FAFAFA",
          border: "#D5D5D5",
          borderStrong: "#B3B3B3",
          ink: "#1B1B1B",
          ink2: "#4B4B4B",
          ink3: "#7A7A7A",
        },
        // graph node type palette — cool, muted, Spectrum-adjacent
        node: {
          pathway: "#2680EB",
          enzyme: "#9256D9",
          molecule: "#2D9D78",
          process: "#E68619",
          structure: "#6E6E6E",
        },
      },
      fontFamily: {
        sans: [
          "Source Sans 3",
          "Source Sans Pro",
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "sans-serif",
        ],
      },
      boxShadow: {
        card: "0 1px 2px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.06)",
        cardHover: "0 2px 4px rgba(0,0,0,0.06), 0 8px 24px rgba(0,0,0,0.10)",
      },
      borderRadius: {
        xl2: "14px",
      },
    },
  },
  plugins: [],
};

export default config;
