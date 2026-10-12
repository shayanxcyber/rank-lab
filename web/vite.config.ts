import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) return;
          if (
            id.includes("@observablehq") ||
            id.includes("/d3-") ||
            id.includes("/internmap/")
          )
            return "plot";
          if (id.includes("@base-ui") || id.includes("@floating-ui"))
            return "controls";
          if (id.includes("motion")) return "motion";
          if (
            id.includes("/react/") ||
            id.includes("/react-dom/") ||
            id.includes("/scheduler/")
          )
            return "react-runtime";
          if (id.includes("@tanstack")) return "table";
        },
      },
    },
  },
});
