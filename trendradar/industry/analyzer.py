# coding=utf-8
"""
行业专项分析器

按 RSS feed type 聚合内容，并调用 AI 生成专项简报。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from trendradar.ai.client import AIClient
from trendradar.ai.prompt_loader import load_prompt_template


@dataclass
class IndustryAnalysisResult:
    success: bool = False
    type: str = ""
    display_name: str = ""
    summary: str = ""
    consensus_topics: str = ""
    fresh_signals: str = ""
    kol_media_divergence: str = ""
    notable_entities: str = ""
    risk_watch: str = ""
    raw_response: str = ""
    raw_count: int = 0
    kol_count: int = 0
    media_count: int = 0
    skipped: bool = False
    error: str = ""


class IndustryAnalyzer:
    """行业专项分析器。"""

    def __init__(
        self,
        ai_config: Dict[str, Any],
        industry_config: Dict[str, Any],
        get_time_func: Callable[[], datetime],
        debug: bool = False,
    ):
        self.ai_config = ai_config
        self.industry_config = industry_config
        self.get_time_func = get_time_func
        self.debug = debug
        self.client = AIClient(ai_config)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").strip().lower()
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _dedup_items(self, items: List[Dict]) -> List[Dict]:
        seen = set()
        deduped = []
        for item in items:
            key = item.get("url") or self._normalize_text(
                f"{item.get('title', '')} {item.get('summary', '')}"
            )
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _filter_by_freshness(self, items: List[Dict], hours: int) -> List[Dict]:
        now = self.get_time_func()
        threshold = now - timedelta(hours=hours)
        filtered = []
        for item in items:
            published_at = item.get("published_at")
            if not published_at:
                filtered.append(item)
                continue
            try:
                published_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                if published_dt.tzinfo is not None:
                    published_dt = published_dt.replace(tzinfo=None)
                if published_dt >= threshold:
                    filtered.append(item)
            except Exception:
                filtered.append(item)
        return filtered

    def _format_item(self, item: Dict) -> str:
        source_kind = item.get("source_kind", "media")
        feed_name = item.get("feed_name", item.get("feed_id", "未知源"))
        lang = item.get("lang_detected", "unknown")
        author = item.get("author", "")
        published_at = item.get("published_at", "")
        title = item.get("title", "")
        summary = item.get("summary", "")
        weight = item.get("weight", 1)
        tags = ", ".join(item.get("tags", []) or [])
        meta = [f"source={feed_name}", f"kind={source_kind}", f"lang={lang}", f"weight={weight}"]
        if author:
            meta.append(f"author={author}")
        if published_at:
            meta.append(f"published={published_at}")
        if tags:
            meta.append(f"tags={tags}")
        line = f"- [{' | '.join(meta)}]\n  title: {title}"
        if summary:
            line += f"\n  summary: {summary}"
        return line

    def _build_prompt(self, group: Dict, items: List[Dict], system_prompt: str, user_template: str) -> tuple[str, str]:
        type_name = group["TYPE"]
        display_name = group["DISPLAY_NAME"]
        output_language = self.industry_config.get("OUTPUT_LANGUAGE", "zh-CN")
        kol_items = [item for item in items if item.get("source_kind") == "kol"][: group["AI"]["MAX_KOL_ITEMS"]]
        media_items = [item for item in items if item.get("source_kind") == "media"][: group["AI"]["MAX_MEDIA_ITEMS"]]
        remaining = max(group["AI"]["MAX_ITEMS"] - len(kol_items) - len(media_items), 0)
        others = [
            item for item in items
            if item.get("source_kind") not in ("kol", "media")
        ][:remaining]
        selected = (kol_items + media_items + others)[: group["AI"]["MAX_ITEMS"]]

        content_lines = [self._format_item(item) for item in selected]
        user_prompt = user_template
        user_prompt = user_prompt.replace("{type}", type_name)
        user_prompt = user_prompt.replace("{display_name}", display_name)
        user_prompt = user_prompt.replace("{output_language}", output_language)
        user_prompt = user_prompt.replace("{current_time}", self.get_time_func().strftime("%Y-%m-%d %H:%M:%S"))
        user_prompt = user_prompt.replace("{total_count}", str(len(items)))
        user_prompt = user_prompt.replace("{kol_count}", str(len([i for i in items if i.get("source_kind") == "kol"])))
        user_prompt = user_prompt.replace("{media_count}", str(len([i for i in items if i.get("source_kind") == "media"])))
        user_prompt = user_prompt.replace("{items_content}", "\n\n".join(content_lines))
        return system_prompt, user_prompt

    def _parse_response(self, response: str, group: Dict) -> IndustryAnalysisResult:
        result = IndustryAnalysisResult(success=False, type=group["TYPE"], display_name=group["DISPLAY_NAME"], raw_response=response)
        try:
            start = response.find("{")
            end = response.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("AI 响应中未找到 JSON")
            data = json.loads(response[start:end + 1])
            result.success = True
            result.summary = data.get("summary", "")
            result.consensus_topics = data.get("consensus_topics", "")
            result.fresh_signals = data.get("fresh_signals", "")
            result.kol_media_divergence = data.get("kol_media_divergence", "")
            result.notable_entities = data.get("notable_entities", "")
            result.risk_watch = data.get("risk_watch", "")
            return result
        except Exception as e:
            result.error = f"行业分析解析失败: {e}"
            return result

    def analyze_group(self, group: Dict, items: List[Dict]) -> IndustryAnalysisResult:
        result = IndustryAnalysisResult(type=group["TYPE"], display_name=group["DISPLAY_NAME"])
        if not group.get("ENABLED", True) or not group.get("AI", {}).get("ENABLED", True):
            result.skipped = True
            result.error = "行业分析已禁用"
            return result

        filtered = items
        freshness = group.get("FRESHNESS", {})
        if freshness.get("ENABLED", True):
            filtered = self._filter_by_freshness(filtered, freshness.get("HOURS", 24))

        source_kinds = set(group.get("SOURCE_KINDS", ["kol", "media"]))
        filtered = [item for item in filtered if item.get("source_kind", "media") in source_kinds]
        filtered = self._dedup_items(filtered)

        result.raw_count = len(filtered)
        result.kol_count = sum(1 for item in filtered if item.get("source_kind") == "kol")
        result.media_count = sum(1 for item in filtered if item.get("source_kind") == "media")

        if not filtered:
            result.skipped = True
            result.error = "该行业分组暂无可分析 RSS 内容"
            return result

        system_prompt, user_template = load_prompt_template(
            group.get("AI", {}).get("PROMPT_FILE", "industry_analysis_prompt.txt"),
            label="INDUSTRY",
        )
        if not user_template:
            result.error = "行业分析提示词为空"
            return result

        system_prompt, user_prompt = self._build_prompt(group, filtered, system_prompt, user_template)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if self.debug:
            print(f"[Industry][DEBUG] type={group['TYPE']} prompt:\n{user_prompt}")

        try:
            response = self.client.chat(messages)
            parsed = self._parse_response(response, group)
            parsed.raw_count = result.raw_count
            parsed.kol_count = result.kol_count
            parsed.media_count = result.media_count
            return parsed
        except Exception as e:
            result.error = f"行业分析失败: {e}"
            return result

    def analyze(self, rss_items: List[Dict]) -> List[IndustryAnalysisResult]:
        groups = self.industry_config.get("GROUPS", [])
        if not self.industry_config.get("ENABLED", False) or not groups:
            return []

        grouped_items: Dict[str, List[Dict]] = {}
        for item in rss_items or []:
            item_type = str(item.get("feed_type", "") or "").strip()
            if not item_type:
                continue
            grouped_items.setdefault(item_type, []).append(item)

        results: List[IndustryAnalysisResult] = []
        for group in groups:
            items = grouped_items.get(group["TYPE"], [])
            results.append(self.analyze_group(group, items))
        return results
