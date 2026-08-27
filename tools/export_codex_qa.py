#!/usr/bin/env python3
"""Export completed user/assistant turns from a Codex rollout JSONL to Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def message_text(payload: dict) -> str:
    parts = []
    for item in payload.get("content", []):
        item_type = item.get("type")
        if item_type in {"input_text", "output_text"}:
            parts.append(item.get("text", ""))
        elif item_type == "input_image":
            parts.append("[图片输入]")
    return "\n\n".join(part for part in parts if part).strip()


def completed_turns(session_path: Path) -> list[dict]:
    turns = []
    pending_users = []
    seen_ids = set()

    with session_path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("type") != "response_item":
                continue
            payload = record.get("payload", {})
            if payload.get("type") != "message":
                continue

            item_id = payload.get("id")
            if item_id and item_id in seen_ids:
                continue
            if item_id:
                seen_ids.add(item_id)

            role = payload.get("role")
            text = message_text(payload)
            if not text:
                continue

            if role == "user":
                # Runtime context is injected by the client, rather than entered by
                # the user, and is intentionally omitted from the conversation log.
                if text.startswith("<environment_context>"):
                    continue
                pending_users.append(
                    {"timestamp": record.get("timestamp", ""), "text": text}
                )
            elif role == "assistant" and payload.get("phase") == "final_answer":
                if not pending_users:
                    continue
                turns.append(
                    {
                        "users": pending_users,
                        "answer_timestamp": record.get("timestamp", ""),
                        "answer": text,
                    }
                )
                pending_users = []

    return turns


def render(turns: list[dict], session_path: Path) -> str:
    lines = [
        "# QA 对话记录",
        "",
        "> 本文件由 Codex 会话记录回填，并在后续对话中持续追加。",
        "> 仅记录用户输入和助手最终回答；不记录内部推理、工具输出及中间进度。",
        f"> 历史来源：`{session_path}`",
        "",
    ]

    for index, turn in enumerate(turns, 1):
        started = turn["users"][0]["timestamp"]
        lines.extend([f"## 第 {index} 轮 — {started}", "", "### 用户输入", ""])
        for user_index, user in enumerate(turn["users"], 1):
            if len(turn["users"]) > 1:
                lines.extend([f"#### 输入 {user_index}", ""])
            lines.extend([user["text"], ""])
        lines.extend(
            [
                "### 助手回答",
                "",
                turn["answer"],
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    turns = completed_turns(args.session)
    args.output.write_text(render(turns, args.session), encoding="utf-8")
    print(f"exported_turns={len(turns)} output={args.output}")


if __name__ == "__main__":
    main()
