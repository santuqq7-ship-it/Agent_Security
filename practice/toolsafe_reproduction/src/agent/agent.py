"""Original ToolSafe Agent_Core with a local dependency boundary.

Source baseline: ``ToolSafe/src/agent/agent.py`` at commit
``46358fa424a927a895c6c8322f99032c4eb5155e``.
"""

from openai import OpenAI
from copy import deepcopy
try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
except ImportError:  # Transformers Agent mode does not require vLLM.
    LLM = None
    SamplingParams = None

import torch
import numpy as np

@torch.no_grad()
def token_entropy(logits):
    """
    logits: [batch, vocab_size]
    return: [batch] entropy
    """
    # 使用 log_softmax 避免数值不稳定
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [batch, vocab]
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)  # [batch]
    
    # 额外安全处理 NaN（如果仍然出现极端值）
    entropy = torch.where(torch.isnan(entropy), torch.zeros_like(entropy), entropy)
    
    return entropy



class Agent_Core:
    def __init__(self, system="", agentic_model=None, trace_callback=None):
        self.system = system
        self.messages = []
        self.trace_callback = trace_callback

        if self.system:
            self.messages.append({"role": "system", "content": system})

        self.agentic_model=agentic_model

    def _emit_trace(self, event):
        """Send an optional, read-only event without changing Agent behavior."""
        if self.trace_callback is not None:
            self.trace_callback({"source": "agent", **event})


    def __call__(self, message, json_format=False):
        if self.messages[-1]["role"] == "user" and self.messages[-1]["content"] == message:
            result = self.execute(json_format)
            self.messages.append({"role": "assistant", "content": result})
        else:
            self.messages.append({"role": "user", "content": message})
            result = self.execute(json_format)
            self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self, json_format):
        # Capture the exact message list before the selected backend generates.
        self._emit_trace({
            "kind": "agent_request",
            "messages": deepcopy(self.messages),
            "json_format": json_format,
        })
        if self.agentic_model.model_type=="api":
            if json_format:
                response = self.agentic_model.llm.chat.completions.create(
                    model=self.agentic_model.model_name,
                    messages=self.messages,
                    response_format={"type": "json_object"}
                )
            else:
                response = self.agentic_model.llm.chat.completions.create(
                    model=self.agentic_model.model_name,
                    messages=self.messages
                )
            response_text = response.choices[0].message.content
            self._emit_trace({"kind": "agent_response", "text": response_text})
            return response_text
        elif self.agentic_model.model_type=="analysis":
            # ===== 构造输入 =====
            inputs = self.agentic_model.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            self._emit_trace({"kind": "agent_rendered_prompt", "prompt": inputs})

            model_inputs = self.agentic_model.tokenizer(
                inputs,
                return_tensors="pt"
            ).to(self.agentic_model.llm.device)

            # ===== generate with scores =====
            outputs = self.agentic_model.llm.generate(
                **model_inputs,
                max_new_tokens=getattr(self.agentic_model, "max_new_tokens", 512),
                do_sample=False,
                return_dict_in_generate=True,   # ⭐ 关键
                output_scores=True              # ⭐ 关键
            )

            # ===== decode text =====
            response = self.agentic_model.tokenizer.decode(
                outputs.sequences[0][model_inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            ).strip()

            # ==========================
            # 🔥 Token-level entropy
            # ==========================
            entropies = []

            for step_logits in outputs.scores:
                # step_logits: [batch, vocab]
                entropy = token_entropy(step_logits)
                entropies.append(entropy.item())

            # ===== 统计你关心的指标 =====
            entropies_np = np.array(entropies)

            self.last_entropies = entropies
            self.last_entropy_stats = {
                "mean": float(entropies_np.mean()) if len(entropies_np) > 0 else 0.0,
                "length": len(entropies_np)
            }
            self._emit_trace({
                "kind": "agent_response",
                "text": response,
                "entropy_stats": self.last_entropy_stats,
            })
            return response
        else:
            response = self.agentic_model.llm.chat(self.messages, sampling_params=self.agentic_model.sampling)
            response_text = response[0].outputs[0].text.strip()
            self._emit_trace({"kind": "agent_response", "text": response_text})
            return response_text
