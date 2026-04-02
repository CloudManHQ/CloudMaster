import { Link } from "react-router-dom";
import { Folder, FileText } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

// This would typically come from an API or generated at build time
const docSections = [
  {
    id: "00",
    title: "AI Introduction",
    description: "Getting started with AI fundamentals",
    docs: [
      { slug: "ai-fundamentals", title: "AI Fundamentals" },
      { slug: "ai-technology-landscape", title: "Technology Landscape" },
      { slug: "ai-history", title: "History Timeline" },
      { slug: "ai-tools", title: "Tools & Practice" },
      { slug: "ai-ethics", title: "Ethics & Society" },
      { slug: "ai-future", title: "Future Trends" },
      { slug: "ai-learning", title: "Learning Resources" },
    ],
  },
  {
    id: "01",
    title: "Fundamentals",
    description: "Mathematical and computational foundations",
    docs: [
      { slug: "linear-algebra", title: "Linear Algebra" },
      { slug: "probability", title: "Probability & Statistics" },
      { slug: "data-structures", title: "Data Structures" },
      { slug: "distributed-systems", title: "Distributed Systems" },
      { slug: "ai-hardware", title: "AI Hardware" },
    ],
  },
  {
    id: "04",
    title: "NLP & LLMs",
    description: "Natural language processing and large language models",
    docs: [
      { slug: "transformer", title: "Transformer Architecture" },
      { slug: "llm-architectures", title: "LLM Architectures" },
      { slug: "fine-tuning", title: "Fine-tuning Techniques" },
      { slug: "prompt-engineering", title: "Prompt Engineering" },
    ],
  },
];

export function DocsPage() {
  return (
    <div className="container py-8">
      <div className="mx-auto max-w-4xl">
        <h1 className="text-3xl font-bold mb-6">Documentation</h1>
        <p className="text-muted-foreground mb-8">
          Browse our comprehensive AI knowledge base organized by topics.
        </p>

        <div className="space-y-6">
          {docSections.map((section) => (
            <Card key={section.id}>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Folder className="h-5 w-5 text-primary" />
                  <CardTitle>{section.title}</CardTitle>
                </div>
                <p className="text-sm text-muted-foreground">
                  {section.description}
                </p>
              </CardHeader>
              <CardContent>
                <ul className="grid gap-2 sm:grid-cols-2">
                  {section.docs.map((doc) => (
                    <li key={doc.slug}>
                      <Link
                        to={`/docs/${section.id}-${doc.slug}`}
                        className="flex items-center gap-2 p-2 rounded-md hover:bg-muted transition-colors"
                      >
                        <FileText className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm">{doc.title}</span>
                      </Link>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
