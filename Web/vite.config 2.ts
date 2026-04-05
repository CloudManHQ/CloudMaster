import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { visualizer } from "rollup-plugin-visualizer";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [
    react(),
    mode === "analyze" &&
      visualizer({
        open: true,
        filename: "dist/stats.html",
      }),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom"],
          router: ["react-router-dom"],
          query: ["@tanstack/react-query"],
          markdown: ["react-markdown", "remark-gfm", "rehype-highlight"],
          fuse: ["fuse.js"],
        },
      },
    },
    sourcemap: true,
  },
  server: {
    port: 3055,
    host: "0.0.0.0",  // Listen on all addresses
    strictPort: true,  // Fail if port is in use
    open: true,
  },
  preview: {
    port: 3055,
    host: "0.0.0.0",
    strictPort: true,
  },
}));
