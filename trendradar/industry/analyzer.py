# coding=utf-8
"""
行业专项分析器

按 RSS feed type 聚合内容，并调用 AI 生成专项简报。
"""

from __future__ import annotations

from collections import Counter
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
    def _format_counter(counter: Counter, max_items: int = 8) -> str:
        if not counter:
            return "-"
        parts = [f"{key}:{value}" for key, value in counter.most_common(max_items)]
        if len(counter) > max_items:
            parts.append(f"...(+{len(counter) - max_items})")
        return ", ".join(parts)

    def _log(self, message: str) -> None:
        print(f"[Industry] {message}")

    @staticmethod
    def _preview_text(text: str, max_len: int = 240) -> str:
        normalized = re.sub(r"\s+", " ", (text or "")).strip()
        if not normalized:
            return "<empty>"
        if len(normalized) > max_len:
            return normalized[:max_len] + "..."
        return normalized

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

    def _select_prompt_items(self, group: Dict, items: List[Dict]) -> List[Dict]:
        kol_items = [item for item in items if item.get("source_kind") == "kol"][: group["AI"]["MAX_KOL_ITEMS"]]
        media_items = [item for item in items if item.get("source_kind") == "media"][: group["AI"]["MAX_MEDIA_ITEMS"]]
        remaining = max(group["AI"]["MAX_ITEMS"] - len(kol_items) - len(media_items), 0)
        others = [
            item for item in items
            if item.get("source_kind") not in ("kol", "media")
        ][:remaining]
        return (kol_items + media_items + others)[: group["AI"]["MAX_ITEMS"]]

    def _build_prompt(self, group: Dict, items: List[Dict], system_prompt: str, user_template: str) -> tuple[str, str]:
        type_name = group["TYPE"]
        display_name = group["DISPLAY_NAME"]
        output_language = self.industry_config.get("OUTPUT_LANGUAGE", "zh-CN")
        selected = self._select_prompt_items(group, items)

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
                preview = self._preview_text(response)
                raise ValueError(
                    f"AI 响应中未找到 JSON(len={len(response or '')}, preview={preview})"
                )
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
        group_name = result.display_name or result.type
        source_kinds = set(group.get("SOURCE_KINDS", ["kol", "media"]))
        freshness = group.get("FRESHNESS", {})
        dedup_enabled = group.get("DEDUP", {}).get("ENABLED", True)

        self._log(
            f"group={group_name} start: input={len(items)}, "
            f"enabled={group.get('ENABLED', True)}, ai_enabled={group.get('AI', {}).get('ENABLED', True)}, "
            f"source_kinds={sorted(source_kinds)}, freshness={freshness.get('ENABLED', True)}/{freshness.get('HOURS', 24)}h, "
            f"dedup={dedup_enabled}"
        )

        if not group.get("ENABLED", True) or not group.get("AI", {}).get("ENABLED", True):
            result.skipped = True
            result.error = (
                f"行业分析已禁用(group.enabled={group.get('ENABLED', True)}, "
                f"group.ai.enabled={group.get('AI', {}).get('ENABLED', True)})"
            )
            self._log(f"group={group_name} skipped: {result.error}")
            return result

        filtered = items
        before_freshness = len(filtered)
        if freshness.get("ENABLED", True):
            filtered = self._filter_by_freshness(filtered, freshness.get("HOURS", 24))
        freshness_removed = before_freshness - len(filtered)
        self._log(
            f"group={group_name} after freshness: kept={len(filtered)}, removed={freshness_removed}"
        )

        before_source_kind = len(filtered)
        source_kind_counter = Counter(str(item.get("source_kind", "media") or "media") for item in filtered)
        filtered = [item for item in filtered if item.get("source_kind", "media") in source_kinds]
        source_kind_removed = before_source_kind - len(filtered)
        self._log(
            f"group={group_name} after source_kind: kept={len(filtered)}, removed={source_kind_removed}, "
            f"source_kind_breakdown={self._format_counter(source_kind_counter)}"
        )

        before_dedup = len(filtered)
        if dedup_enabled:
            filtered = self._dedup_items(filtered)
        dedup_removed = before_dedup - len(filtered)
        self._log(
            f"group={group_name} after dedup: kept={len(filtered)}, removed={dedup_removed}, "
            f"feeds={self._format_counter(Counter(item.get('feed_name', item.get('feed_id', 'unknown')) for item in filtered))}"
        )

        result.raw_count = len(filtered)
        result.kol_count = sum(1 for item in filtered if item.get("source_kind") == "kol")
        result.media_count = sum(1 for item in filtered if item.get("source_kind") == "media")

        if not filtered:
            result.skipped = True
            result.error = (
                "该行业分组暂无可分析 RSS 内容"
                f"(input={len(items)}, freshness_kept={before_freshness - freshness_removed}, "
                f"source_kind_kept={before_source_kind - source_kind_removed}, dedup_kept={len(filtered)})"
            )
            self._log(f"group={group_name} skipped: {result.error}")
            return result

        system_prompt, user_template = load_prompt_template(
            group.get("AI", {}).get("PROMPT_FILE", "industry_analysis_prompt.txt"),
            label="INDUSTRY",
        )
        if not user_template:
            result.error = "行业分析提示词为空"
            self._log(f"group={group_name} failed before AI: {result.error}")
            return result

        selected_items = self._select_prompt_items(group, filtered)
        self._log(
            f"group={group_name} prompt items: selected={len(selected_items)}/{len(filtered)}, "
            f"kol={sum(1 for item in selected_items if item.get('source_kind') == 'kol')}, "
            f"media={sum(1 for item in selected_items if item.get('source_kind') == 'media')}, "
            f"max_items={group['AI']['MAX_ITEMS']}"
        )

        system_prompt, user_prompt = self._build_prompt(group, filtered, system_prompt, user_template)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if self.debug:
            print(f"[Industry][DEBUG] type={group['TYPE']} prompt:\n{user_prompt}")

        try:
            response = self.client.chat(messages)
            self._log(
                f"group={group_name} raw response: len={len(response or '')}, "
                f"preview={self._preview_text(response)}"
            )
            parsed = self._parse_response(response, group)
            parsed.raw_count = result.raw_count
            parsed.kol_count = result.kol_count
            parsed.media_count = result.media_count
            if parsed.success:
                self._log(
                    f"group={group_name} success: raw_count={parsed.raw_count}, "
                    f"kol={parsed.kol_count}, media={parsed.media_count}"
                )
            else:
                self._log(f"group={group_name} failed after AI: {parsed.error}")
            return parsed
        except Exception as e:
            result.error = f"行业分析失败: {e}"
            self._log(f"group={group_name} failed during AI call: {result.error}")
            return result

    def analyze(self, rss_items: List[Dict]) -> List[IndustryAnalysisResult]:
        groups = self.industry_config.get("GROUPS", [])
        if not self.industry_config.get("ENABLED", False) or not groups:
            self._log(
                f"skip analyze: enabled={self.industry_config.get('ENABLED', False)}, groups={len(groups)}"
            )
            return []

        grouped_items: Dict[str, List[Dict]] = {}
        missing_type_count = 0
        for item in rss_items or []:
            item_type = str(item.get("feed_type", "") or "").strip()
            if not item_type:
                missing_type_count += 1
                continue
            grouped_items.setdefault(item_type, []).append(item)

        self._log(
            f"input summary: rss_items={len(rss_items or [])}, with_type={sum(len(items) for items in grouped_items.values())}, "
            f"without_type={missing_type_count}, type_breakdown={self._format_counter(Counter({key: len(value) for key, value in grouped_items.items()}))}"
        )

        results: List[IndustryAnalysisResult] = []
        for group in groups:
            items = grouped_items.get(group["TYPE"], [])
            self._log(
                f"group mapping: type={group['TYPE']}, matched_items={len(items)}, display_name={group['DISPLAY_NAME']}"
            )
            results.append(self.analyze_group(group, items))
        return results
