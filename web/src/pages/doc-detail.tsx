import { useParams, Link } from "react-router-dom";
import { ArrowLeft, ArrowRight, Clock, Calendar } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

// Mock markdown content - in production this would be fetched from the docs
const mockContent = `# AI Fundamentals

> **One sentence summary**: Artificial Intelligence is the science of making machines simulate human intelligence—from recognizing cat photos to understanding natural language.

## What is AI?

### Definition and Essence

Artificial Intelligence (AI) is the field of computer science focused on creating systems capable of performing tasks that typically require human intelligence.

\`\`\`
AI Capabilities:
├── Perception: Seeing, hearing, sensing
├── Reasoning: Logic, inference, problem-solving
├── Learning: From data and experience
├── Decision-making: Choosing optimal actions
└── Language: Understanding and generation
\`\`\`

### AI vs Traditional Programming

| Aspect | Traditional Programming | AI/Machine Learning |
|--------|------------------------|---------------------|
| Approach | Human writes rules → Machine executes | Human provides data → Machine learns rules |
| Example | IF contains "prize" THEN spam | Learn from 1M emails → Auto-detect spam |

## The Three Types of AI

### 1. Artificial Narrow Intelligence (ANI)
- **Current stage**
- Excels at specific tasks
- Examples: AlphaGo, Siri, recommendation systems

### 2. Artificial General Intelligence (AGI)
- **Future goal**
- Human-level general intelligence
- Can learn any intellectual task

### 3. Artificial Super Intelligence (ASI)
- **Theoretical**
- Beyond human intelligence
- Unknown timeline

## Core Technologies

- **Machine Learning**: Learning patterns from data
- **Deep Learning**: Neural networks with many layers
- **Natural Language Processing**: Understanding human language
- **Computer Vision**: Interpreting visual information

## Applications

AI is used in:
- Healthcare (diagnosis, drug discovery)
- Finance (fraud detection, trading)
- Transportation (autonomous vehicles)
- Education (personalized learning)
- Entertainment (content recommendation)

---

*Last updated: 2026-04-01*
`;

// Mock navigation - in production this would be dynamic
const navigation = {
  prev: { slug: "00-ai-history", title: "History Timeline" },
  next: { slug: "00-ai-technology", title: "Technology Landscape" },
};

export function DocDetailPage() {
  const { slug } = useParams<{ slug: string }>();

  // In production, fetch the actual markdown content based on slug
  console.log("Loading doc:", slug);

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
        </div>

        {/* Content */}
        <Card className="p-8">
          <article className="markdown-content">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
            >
              {mockContent}
            </ReactMarkdown>
          </article>
        </Card>

        {/* Footer */}
        <div className="mt-8 flex items-center justify-between text-sm text-muted-foreground">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <Calendar className="h-4 w-4" />
              Updated: 2026-04-01
            </span>
            <span className="flex items-center gap-1">
              <Clock className="h-4 w-4" />
              Read time: 5 min
            </span>
          </div>
        </div>

        {/* Navigation */}
        <div className="mt-8 grid gap-4 sm:grid-cols-2">
          {navigation.prev && (
            <Button variant="outline" asChild className="justify-start">
              <Link to={`/docs/${navigation.prev.slug}`}>
                <ArrowLeft className="mr-2 h-4 w-4" />
                <div className="flex flex-col items-start">
                  <span className="text-xs text-muted-foreground">Previous</span>
                  <span className="text-sm font-medium">{navigation.prev.title}</span>
                </div>
              </Link>
            </Button>
          )}
          {navigation.next && (
            <Button variant="outline" asChild className="justify-end sm:justify-end">
              <Link to={`/docs/${navigation.next.slug}`}>
                <div className="flex flex-col items-end">
                  <span className="text-xs text-muted-foreground">Next</span>
                  <span className="text-sm font-medium">{navigation.next.title}</span>
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
