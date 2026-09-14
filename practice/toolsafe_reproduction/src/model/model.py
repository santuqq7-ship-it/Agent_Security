"""ToolSafe model adapters used by the local Transformers reproduction.

Source baseline: ``ToolSafe/src/model/model.py`` at commit
``46358fa424a927a895c6c8322f99032c4eb5155e``.  The original file is kept
unchanged; this overlay only makes the dependency boundary explicit and adds
the local Transformers Guardian path needed on macOS.
"""

from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy

try:
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
except ImportError:  # Local Transformers mode does not require vLLM.
    LLM = None
    SamplingParams = None

# Keep PEFT optional for the original base-model inference path.  The adapter
# path imports it lazily unless tests or callers provide this symbol directly.
PeftModel = None

import torch
from agent.agent_prompts import *
from utils.guardian_parser import *

import httpx
user = "xxx"
pwd  = "xxx"
auth = httpx.BasicAuth(user, pwd)

class Model:
    def __init__(self, model_name="gpt-4o", model_path="", model_type="api", api_base="", api_key ="", max_new_tokens=512):
        self.model_name = model_name
        self.model_type = model_type
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.trace_callback = None

        if self.model_type == "api":
            self.llm = OpenAI(
                api_key=api_key,
                base_url=api_base,
            )
        elif self.model_type == "analysis":
            device = _resolve_transformers_device()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )

            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
                dtype=_dtype_for_device(device),
            )
            self.llm.to(device)
            self.llm.eval()

        else:
            if LLM is None or SamplingParams is None:
                raise RuntimeError(
                    "model_type='local' requires vLLM. Install vLLM on a "
                    "Linux/NVIDIA environment or use model_type='analysis'."
                )
            self.llm = LLM(model=model_path)
            self.sampling = SamplingParams(max_tokens=2048, temperature=0.1, top_p=0.9)



class Guardian:
    def __init__(self, model_name="gpt-4o", model_path="", model_type="api", api_base="", api_key ="", trace_callback=None, max_new_tokens=2048, adapter_path=None):
        self.model_name = model_name
        self.model_type = model_type
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.trace_callback = trace_callback
        self.max_new_tokens = max_new_tokens

        if self.model_type == "api":
            if "gpt" in self.model_name.lower() or "claude" in self.model_name.lower() or "gemini" in self.model_name.lower():
                self.llm = OpenAI(
                    api_key=api_key,
                    base_url=api_base
                )
            else:
                self.llm = OpenAI(
                    api_key=api_key,
                    base_url=api_base,
                    http_client=httpx.Client(auth=auth, verify=True)
                )
        elif self.model_type == "analysis":
            device = _resolve_transformers_device()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
                dtype=_dtype_for_device(device),
            )
            if adapter_path:
                peft_model = PeftModel
                if peft_model is None:
                    try:
                        from peft import PeftModel as peft_model
                    except ImportError as exc:
                        raise RuntimeError(
                            "adapter_path was provided, but PEFT is unavailable. "
                            "Install practice/toolsafe_reproduction/sft/requirements-sft.txt."
                        ) from exc
                self.llm = peft_model.from_pretrained(
                    base_model,
                    adapter_path,
                    is_trainable=False,
                )
            else:
                self.llm = base_model
            self.llm.to(device)
            self.llm.eval()

        else:
            if LLM is None or SamplingParams is None:
                raise RuntimeError(
                    "Guardian model_type='local' requires vLLM. Install vLLM "
                    "on a Linux/NVIDIA environment or use 'analysis'."
                )
            self.llm = LLM(model=model_path)
            self.sampling = SamplingParams(max_tokens=2048, temperature=0.1, top_p=0.9)

    def _emit_trace(self, event):
        """Send an optional read-only event for prompt/output inspection."""
        if getattr(self, "trace_callback", None) is not None:
            self.trace_callback({"source": "guardian", **event})

    def _generate_text(self, messages, response_format=None):
        """Generate one assistant text response using the selected backend.

        Transformers uses the same chat template as the Agent and decodes only
        newly generated tokens. API and vLLM branches retain ToolSafe's native
        interfaces so the overlay can still be used on a supported host.
        """
        if self.model_type == "api":
            self._emit_trace({
                "kind": "guardian_request",
                "messages": deepcopy(messages),
                "response_format": response_format,
            })
            kwargs = {"model": self.model_name, "messages": messages}
            if response_format is not None:
                kwargs["response_format"] = response_format
            response = self.llm.chat.completions.create(**kwargs)
            response_text = response.choices[0].message.content
            self._emit_trace({"kind": "guardian_response", "text": response_text})
            return response_text

        if self.model_type == "local":
            self._emit_trace({
                "kind": "guardian_request",
                "messages": deepcopy(messages),
                "response_format": response_format,
            })
            response = self.llm.chat(messages, sampling_params=self.sampling)
            response_text = response[0].outputs[0].text.strip()
            self._emit_trace({"kind": "guardian_response", "text": response_text})
            return response_text

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(prompt, return_tensors="pt")
        model_inputs = model_inputs.to(self.llm.device)
        self._emit_trace({
            "kind": "guardian_request",
            "messages": deepcopy(messages),
            "rendered_prompt": prompt,
            "response_format": response_format,
        })
        outputs = self.llm.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=False,
        )
        prompt_length = model_inputs["input_ids"].shape[-1]
        response_text = self.tokenizer.decode(
            outputs.sequences[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()
        self._emit_trace({"kind": "guardian_response", "text": response_text})
        return response_text


    def get_judgment_res(self, meta_info, max_turn=3):
        guard_input = GUARD_TEMPLATES[self.model_name].format(**meta_info)
        guard_messages = [{"role": "user", "content": guard_input}]
        results = {}
        j = 0
        while(j<max_turn):
            j+=1
            if self.model_type == "api":
                print("=================================")
                print(self.model_name)
                guard_res = self._generate_text(guard_messages)
            else:
                guard_res = self._generate_text(guard_messages)

            if self.model_name == "ashell-guardian" or self.model_name == "TS-Guard" or self.model_name == "qwen2.5-7b-instruct" or self.model_name == "qwen3-8b" or "gpt" in self.model_name:
                parser_res, results = guardian_paser_map[self.model_name](guard_res)
            elif self.model_name == "ashell-guardian-sftonly": #or self.model_name == "qwen2.5-7b-instruct"
                parser_res, results = guardian_paser_map[self.model_name](guard_res)
            else:
                parser_res = guardian_paser_map[self.model_name](guard_res)

            self._emit_trace({
                "kind": "guardian_parse",
                "parser_result": parser_res,
                "results": deepcopy(results),
            })

            if parser_res in [0,1,0.5]:
                return {"risk rating": parser_res, "results": results, "reason": guard_res}
            else:
                continue
        
        return {"reason": guard_res}
    
    def call_tool(self, tool_name, arguments):
        if tool_name == "tool_safety_guardian":
            guardian_res = self.tool_safety_guardian(**arguments)
            return guardian_res

        elif tool_name == "alignment_check":
            alignment_res = self.alignment_check(**arguments)
            return alignment_res
        
        else:
            return "tool name error, the correct name is tool_safety_guardian"

    
    def tool_safety_guardian(self, 
                     user_request, 
                     interaction_history, 
                     current_action, 
                     current_action_description):
        guard_input = GUARD_TEMPLATES[self.model_name].format(env_info=current_action_description,
                                                              user_request=user_request,
                                                              agent_action={"interaction_history": interaction_history,
                                                                            "current_action": current_action})

        guard_messages = [{"role": "user", "content": guard_input}]
        results = {}

        j = 0
        while(j<3):
            j+=1
            guard_res = self._generate_text(guard_messages)

            if self.model_name == "ashell-guardian" or self.model_name == "TS-Guard": #or self.model_name == "qwen2.5-7b-instruct"
                parser_res, results = guardian_paser_map[self.model_name](guard_res)
            else:
                parser_res = guardian_paser_map[self.model_name](guard_res)

            self._emit_trace({
                "kind": "guardian_parse",
                "parser_result": parser_res,
                "results": deepcopy(results),
            })

            if parser_res in [0,1,0.5]:
                return {"risk rating": parser_res, "results": results, "reason": guard_res}
            else:
                continue

        return {"reason": guard_res}
    

    def alignment_check(self, 
                     user_request, 
                     interaction_history, 
                     current_action, 
                     current_action_description):
        system_prompt, user_prompt = GUARD_TEMPLATES["alignmentcheck"][0], GUARD_TEMPLATES["alignmentcheck"][1].format(env_info=current_action_description,
                                                              user_request=user_request,
                                                              agent_action={"interaction_history": interaction_history,
                                                                            "current_action": current_action})

        guard_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        j = 0
        while(j<3):
            j+=1
            try:
                guard_res = self._generate_text(
                    guard_messages,
                    response_format={"type": "json_object"} if self.model_type == "api" else None,
                )

                print("=================================")
                print(guard_res)
                print("=================================")
                parser_res = alignment_check_parser(guard_res)

                self._emit_trace({
                    "kind": "alignment_parse",
                    "parser_result": parser_res,
                })

                return {"alignment_check_passed": not parser_res, "reason": guard_res}
            except Exception as e:
                print(f"Alignment check error: {str(e)}")
                continue

        return {"alignment_check_passed": not parser_res, "reason": guard_res}


def _resolve_transformers_device():
    """Choose the local accelerator available to Transformers."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dtype_for_device(device):
    """Use accelerator-native low precision and stable fp32 on CPU."""
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float16 if device.type == "mps" else torch.float32
