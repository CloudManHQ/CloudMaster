import { useParams, Link } from "react-router-dom";
import { useState, useEffect } from "react";
import { ArrowLeft, ArrowRight, Clock, Calendar, Loader2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import { findDocByFullSlug, getNavigation } from "@/data/docMap";

export function DocDetailPage() {
  const { slug } = useParams<{ slug: string }>();
  const [content, setContent] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const doc = slug ? findDocByFullSlug(slug) : undefined;
  const nav = slug ? getNavigation(slug) : { prev: undefined, next: undefined };

  useEffect(() => {
    if (!slug || !doc) {
      setLoading(false);
      setError("Document not found");
      return;
    }

    setLoading(true);
    setError(null);

    // Fetch the real markdown file from the project root via Vite's fs.allow
    // In dev mode, Vite serves parent directory files when fs.allow includes ".."
    const filePath = `/../${doc.filePath}`;

    fetch(filePath)
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load: ${res.status}`);
        return res.text();
      })
      .then((text) => {
        // Verify it's actually markdown content (not an HTML error page)
        if (text.trim().startsWith("<!DOCTYPE") || text.trim().startsWith("<html")) {
          throw new Error("Received HTML instead of markdown");
        }
        setContent(text);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error loading doc:", err);
        setError(`无法加载文档: ${doc.filePath}`);
        setLoading(false);
      });
  }, [slug, doc]);

  // Estimate read time (~200 words per minute for technical content)
  const wordCount = content.split(/\s+/).length;
  const readTime = Math.max(1, Math.ceil(wordCount / 200));

  const prevSlug = nav.prev ? `${nav.prev.categoryId}-${nav.prev.slug}` : null;
  const nextSlug = nav.next ? `${nav.next.categoryId}-${nav.next.slug}` : null;

  return (
    <div className="container py-8">
      <div className="mx-auto max-w-4xl">
        {/* Breadcrumb */}
        <div className="mb-6">
          <Button variant="ghost" size="sm" asChild className="mb-4">
            <Link to="/docs">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Docs
            </Link>
          </Button>
          {doc && (
            <div className="text-sm text-muted-foreground">
              <span>{doc.category}</span>
              <span className="mx-2">/</span>
              <span className="font-medium text-foreground">{doc.title}</span>
            </div>
          )}
        </div>

        {/* Content */}
        <Card className="p-8">
          {loading && (
            <div className="flex items-center justify-center py-20">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              <span className="ml-3 text-muted-foreground">Loading document...</span>
            </div>
          )}

          {error && (
            <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
              <AlertCircle className="h-10 w-10 mb-4 text-destructive" />
              <p className="text-lg font-medium">{error}</p>
              <p className="text-sm mt-2">Please check the file path or try another document.</p>
            </div>
          )}

          {!loading && !error && (
            <article className="markdown-content prose prose-neutral dark:prose-invert max-w-none">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                rehypePlugins={[rehypeHighlight]}
              >
                {content}
              </ReactMarkdown>
            </article>
          )}
        </Card>

        {/* Footer */}
        {!loading && !error && (
          <div className="mt-8 flex items-center justify-between text-sm text-muted-foreground">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1">
                <Calendar className="h-4 w-4" />
                Updated: 2026-04-03
              </span>
              <span className="flex items-center gap-1">
                <Clock className="h-4 w-4" />
                Read time: {readTime} min
              </span>
            </div>
          </div>
        )}

        {/* Navigation */}
        <div className="mt-8 grid gap-4 sm:grid-cols-2">
          {nav.prev && prevSlug && (
            <Button variant="outline" asChild className="justify-start">
              <Link to={`/docs/${prevSlug}`}>
                <ArrowLeft className="mr-2 h-4 w-4" />
                <div className="flex flex-col items-start">
                  <span className="text-xs text-muted-foreground">Previous</span>
                  <span className="text-sm font-medium">{nav.prev.title}</span>
                </div>
              </Link>
            </Button>
          )}
          {nav.next && nextSlug && (
            <Button variant="outline" asChild className="justify-end sm:col-start-2">
              <Link to={`/docs/${nextSlug}`}>
                <div className="flex flex-col items-end">
                  <span className="text-xs text-muted-foreground">Next</span>
                  <span className="text-sm font-medium">{nav.next.title}</span>
                </div>
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
