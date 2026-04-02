import { Routes, Route } from "react-router-dom";
import { MainLayout } from "@/components/layout/main-layout";
import { HomePage } from "@/pages/home";
import { DocsPage } from "@/pages/docs";
import { DocDetailPage } from "@/pages/doc-detail";
import { SearchPage } from "@/pages/search";
import { NotFoundPage } from "@/pages/not-found";

export function App() {
  return (
    <MainLayout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/docs" element={<DocsPage />} />
        <Route path="/docs/:slug" element={<DocDetailPage />} />
        <Route path="/search" element={<SearchPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </MainLayout>
  );
}
