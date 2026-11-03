#!/usr/bin/env python3

# OpenAI-compatible backend (vLLM, llama.cpp server, ...). Uses only the standard library so it
# imports in any environment. Unlike OpenAIChat it sends max_tokens / temperature / stop / extra_body
# and records, for every call, the raw response, token usage, latency and finish reason.

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf

from habitat_llm.llm.base_llm import BaseLLM, Prompt


class OpenAICompat(BaseLLM):
    def __init__(self, conf: DictConfig):
        self.llm_conf = conf
        self.generation_params = conf.generation_params
        self.base_url = str(conf.base_url).rstrip("/")
        self.api_key = str(conf.get("api_key", "EMPTY"))
        self.extra_body: Dict[str, Any] = (
            OmegaConf.to_object(conf.extra_body) if conf.get("extra_body") else {}
        )
        self.reasoning_effort = conf.get("reasoning_effort", None)
        self.enable_thinking = conf.get("enable_thinking", None)
        self.timeout = float(conf.get("request_timeout", 900))
        self.retries = int(conf.get("retries", 3))
        self.call_log_path = conf.get("call_log_path", None)
        self.verbose = bool(conf.get("verbose", False))
        # set by the runner so every record can be joined to an episode/agent
        self.context: Dict[str, Any] = {}
        self.call_count = 0
        self.last_call: Optional[Dict[str, Any]] = None
        self._prev_prompt = ""

    def _post(self, body: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        last_err = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after {self.retries} attempts: {last_err}")

    def generate(
        self,
        prompt: Prompt,
        stop: Optional[str] = None,
        max_length: Optional[int] = None,
        generation_args=None,
        request_timeout: Optional[int] = None,
    ):
        if not isinstance(prompt, str):
            raise ValueError("OpenAICompat supports text prompts only")
        gp = OmegaConf.to_object(self.generation_params)
        if stop is None and gp.get("stop"):
            stop = gp["stop"]
        body: Dict[str, Any] = {
            "model": gp["model"],
            "messages": [
                {"role": "system", "content": self.llm_conf.system_message},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": int(max_length if max_length is not None else gp.get("max_tokens", 250)),
            "temperature": float(gp.get("temperature", 0)),
            "top_p": float(gp.get("top_p", 1)),
        }
        if stop:
            body["stop"] = [stop] if isinstance(stop, str) else list(stop)
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": bool(self.enable_thinking)}
        body.update(self.extra_body)

        t0 = time.time()
        out = self._post(body)
        latency = time.time() - t0
        choice = out["choices"][0]
        text = choice["message"].get("content") or ""
        usage = out.get("usage", {})
        shared = len(os.path.commonprefix([self._prev_prompt, prompt]))
        record = {
            "ts": t0,
            "call_idx": self.call_count,
            "model": body["model"],
            "latency_s": round(latency, 4),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_chars": len(prompt),
            # prompt = previous_prompt[:prompt_shared_prefix_chars] + prompt_delta
            "prompt_shared_prefix_chars": shared,
            "prompt_delta": prompt[shared:],
            "stop": body.get("stop"),
            "max_tokens": body["max_tokens"],
            "temperature": body["temperature"],
            "response_raw": text,
            "reasoning": choice["message"].get("reasoning_content") or choice["message"].get("reasoning"),
            **self.context,
        }
        self.call_count += 1
        self.last_call = record
        self._prev_prompt = prompt
        if self.call_log_path:
            with open(self.call_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        if self.verbose:
            print(f"[llm] {latency:.1f}s {usage}")
        if stop and isinstance(stop, str):
            text = text.split(stop)[0]
        return text
