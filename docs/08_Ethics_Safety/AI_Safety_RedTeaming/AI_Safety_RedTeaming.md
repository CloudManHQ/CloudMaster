# AI 安全与红队 (AI Safety & Red Teaming)

通过对抗性测试与防御机制确保 AI 系统的可靠性。

## 1. 核心风险类别 (Risk Categories)

### 提示词注入 (Prompt Injection)
- **描述**: 攻击者通过特定指令绕过模型的安全限制。
- **术语**: 越狱攻击 (Jailbreaking)、间接注入。

### 幻觉与偏见 (Hallucination & Bias)
- **防御**: 检索增强生成 (RAG)、置信度评估。

## 2. 安全评估方法
- **红队测试 (Red Teaming)**: 模拟攻击者行为主动寻找系统漏洞。
- **护栏系统 (Guardrails)**: Llama Guard, NeMo Guardrails。
- **来源**: [OpenAI: Red Teaming Network](https://openai.com/blog/red-teaming-network)

## 3. 来源参考
- [Anthropic: Model Written Evaluations for AI Safety](https://www.anthropic.com/research/model-written-evaluations-for-ai-safety)
