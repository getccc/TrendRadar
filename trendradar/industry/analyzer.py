# coding=utf-8
"""
行业专项分析器

按 RSS feed type 聚合内容，基于 feed tags 构建候选板块，并调用 AI 生成结构化行业报告。
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from trendradar.ai.client import AIClient
from trendradar.ai.prompt_loader import load_prompt_template


@dataclass
class IndustrySectionReference:
    id: int
    title: str = ""
    source_name: str = ""
    url: str = ""
    published_at: str = ""
    source_kind: str = "media"


@dataclass
class IndustrySection:
    key: str
    title: str
    kind: str = "tag_cluster"
    summary: str = ""
    reference_ids: List[int] = field(default_factory=list)
    references: List[IndustrySectionReference] = field(default_factory=list)


@dataclass
class IndustrySummarySection:
    title: str
    summary: str


@dataclass
class IndustryStandaloneReport:
    success: bool = False
    type: str = ""
    display_name: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)
    sections: List[IndustrySection] = field(default_factory=list)
    final_summary: str = ""
    raw_response: str = ""
    error: str = ""


@dataclass
class IndustrySummaryReport:
    type: str = ""
    display_name: str = ""
    summary_sections: List[IndustrySummarySection] = field(default_factory=list)
    final_summary: str = ""


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
    standalone_report: Optional[IndustryStandaloneReport] = None
    summary_report: Optional[IndustrySummaryReport] = None
    include_in_main_report: bool = True
    standalone_feishu_webhook_url: str = ""
    send_standalone_report: bool = False


class IndustryAnalyzer:
    """行业专项分析器。"""

    SPECIAL_SECTION_KEY = "kol-media-divergence"

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

    @staticmethod
    def _slugify(text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fa5]+", "-", (text or "").strip().lower())
        slug = re.sub(r"-+", "-", slug).strip("-")
        return slug or "section"

    @staticmethod
    def _first_sentence(text: str, max_len: int = 120) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        if not cleaned:
            return ""
        parts = re.split(r"(?<=[。！？!?\.])\s+", cleaned, maxsplit=1)
        sentence = parts[0].strip()
        if len(sentence) > max_len:
            return sentence[:max_len].rstrip() + "..."
        return sentence

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

    def _select_prompt_items(self, group: Dict, items: List[Dict]) -> List[Dict]:
        kol_items = [item for item in items if item.get("source_kind") == "kol"][: group["AI"]["MAX_KOL_ITEMS"]]
        media_items = [item for item in items if item.get("source_kind") == "media"][: group["AI"]["MAX_MEDIA_ITEMS"]]
        remaining = max(group["AI"]["MAX_ITEMS"] - len(kol_items) - len(media_items), 0)
        others = [
            item for item in items
            if item.get("source_kind") not in ("kol", "media")
        ][:remaining]
        return (kol_items + media_items + others)[: group["AI"]["MAX_ITEMS"]]

    def _annotate_items_with_ids(self, items: List[Dict]) -> List[Dict]:
        annotated = []
        for index, item in enumerate(items, start=1):
            cloned = dict(item)
            cloned["industry_item_id"] = index
            annotated.append(cloned)
        return annotated

    def _build_candidate_sections(self, group: Dict, items: List[Dict]) -> List[Dict[str, Any]]:
        taxonomy = group.get("REPORT", {}).get("TAG_TAXONOMY", {}) or {}
        strategy = group.get("REPORT", {}).get("SECTION_STRATEGY", {}) or {}
        prefer_feed_tags = strategy.get("PREFER_FEED_TAGS", True)

        buckets: Dict[str, Dict[str, Any]] = {}
        for item in items:
            raw_tags = [str(tag).strip() for tag in (item.get("tags") or []) if str(tag).strip()]
            mapped_titles = []
            for raw_tag in raw_tags:
                mapped_title = taxonomy.get(raw_tag, raw_tag if prefer_feed_tags else "")
                if mapped_title:
                    mapped_titles.append(mapped_title)
            if not mapped_titles:
                mapped_titles = ["其他主题"]

            seen_titles = []
            for mapped_title in mapped_titles:
                if mapped_title not in seen_titles:
                    seen_titles.append(mapped_title)

            for mapped_title in seen_titles[:3]:
                bucket_key = self._slugify(mapped_title)
                bucket = buckets.setdefault(bucket_key, {
                    "bucket_key": bucket_key,
                    "bucket_labels": [],
                    "suggested_title": mapped_title,
                    "item_ids": [],
                    "kol_count": 0,
                    "media_count": 0,
                    "items": [],
                })
                bucket["bucket_labels"].extend([tag for tag in raw_tags if taxonomy.get(tag, tag) == mapped_title])
                bucket["item_ids"].append(item["industry_item_id"])
                bucket["items"].append(item)
                if item.get("source_kind") == "kol":
                    bucket["kol_count"] += 1
                elif item.get("source_kind") == "media":
                    bucket["media_count"] += 1

        normalized = []
        for bucket in buckets.values():
            labels_counter = Counter(bucket["bucket_labels"])
            bucket["bucket_labels"] = [key for key, _ in labels_counter.most_common(6)]
            bucket["item_ids"] = sorted(set(bucket["item_ids"]))
            normalized.append(bucket)

        normalized.sort(key=lambda x: (-len(x["item_ids"]), -x["media_count"] - x["kol_count"], x["suggested_title"]))
        return normalized

    def _format_candidate_section(self, section: Dict[str, Any]) -> str:
        labels = ", ".join(section.get("bucket_labels", [])) or "-"
        return (
            f"- bucket_key={section['bucket_key']} | suggested_title={section['suggested_title']} | "
            f"labels=[{labels}] | item_ids={section['item_ids']} | "
            f"kol={section['kol_count']} | media={section['media_count']}"
        )

    def _format_prompt_item(self, item: Dict) -> str:
        tags = ", ".join(item.get("tags", []) or [])
        return (
            f"[{item['industry_item_id']}] source={item.get('feed_name', item.get('feed_id', '未知源'))} | "
            f"kind={item.get('source_kind', 'media')} | tags={tags or '-'} | "
            f"published={item.get('published_at', '-') or '-'}\n"
            f"title: {item.get('title', '')}\n"
            f"summary: {item.get('summary', '') or '-'}"
        )

    def _build_structured_prompt(
        self,
        group: Dict,
        items: List[Dict],
        candidate_sections: List[Dict[str, Any]],
        system_prompt: str,
        user_template: str,
    ) -> tuple[str, str]:
        display_name = group["DISPLAY_NAME"]
        strategy = group.get("REPORT", {}).get("SECTION_STRATEGY", {}) or {}
        output_language = self.industry_config.get("OUTPUT_LANGUAGE", "zh-CN")
        selected = self._select_prompt_items(group, items)
        selected_ids = {item["industry_item_id"] for item in selected}
        prompt_sections = []
        for candidate in candidate_sections:
            filtered_ids = [item_id for item_id in candidate["item_ids"] if item_id in selected_ids]
            if not filtered_ids:
                continue
            prompt_sections.append({**candidate, "item_ids": filtered_ids})

        if not prompt_sections:
            prompt_sections = candidate_sections[: strategy.get("MAX_SECTIONS", 6)]

        items_content = "\n\n".join(self._format_prompt_item(item) for item in selected)
        candidate_content = "\n".join(self._format_candidate_section(section) for section in prompt_sections)
        instructions = (
            "Return valid JSON only with keys: sections, final_summary. "
            f"sections length must be between {strategy.get('MIN_SECTIONS', 3)} and {strategy.get('MAX_SECTIONS', 6)} when enough evidence exists. "
            f"Each section needs key, title, kind, summary, reference_ids. "
            f"reference_ids should prefer {strategy.get('PREFERRED_REFS_PER_SECTION', 5)} items and stay between "
            f"{strategy.get('MIN_REFS_PER_SECTION', 3)} and {strategy.get('MAX_REFS_PER_SECTION', 8)} when possible. "
            "Do not invent reference IDs. Only use IDs from the provided news items. "
            "kind should be tag_cluster for normal sections or special for KOL/media divergence. "
        )
        if strategy.get("INCLUDE_KOL_MEDIA_DIVERGENCE", True):
            instructions += "Include a KOL/media divergence special section only when a real divergence is supported by evidence. "
        if not strategy.get("FORCE_KOL_MEDIA_DIVERGENCE", False):
            instructions += "Do not force a KOL/media divergence section if evidence is weak. "
        if strategy.get("INCLUDE_FINAL_SUMMARY", True):
            instructions += "final_summary is required. "

        user_prompt = user_template
        replacements = {
            "{type}": group["TYPE"],
            "{display_name}": display_name,
            "{output_language}": output_language,
            "{current_time}": self.get_time_func().strftime("%Y-%m-%d %H:%M:%S"),
            "{total_count}": str(len(items)),
            "{kol_count}": str(sum(1 for i in items if i.get("source_kind") == "kol")),
            "{media_count}": str(sum(1 for i in items if i.get("source_kind") == "media")),
            "{candidate_sections}": candidate_content,
            "{items_content}": items_content,
            "{structured_output_instructions}": instructions,
        }
        for key, value in replacements.items():
            user_prompt = user_prompt.replace(key, value)

        return system_prompt, user_prompt

    def _build_fallback_sections(self, group: Dict, items: List[Dict], candidate_sections: List[Dict[str, Any]]) -> List[IndustrySection]:
        strategy = group.get("REPORT", {}).get("SECTION_STRATEGY", {}) or {}
        max_sections = strategy.get("MAX_SECTIONS", 6)
        preferred_refs = strategy.get("PREFERRED_REFS_PER_SECTION", 5)
        sections = []
        for candidate in candidate_sections[:max_sections]:
            refs = [
                IndustrySectionReference(
                    id=item["industry_item_id"],
                    title=item.get("title", ""),
                    source_name=item.get("feed_name", item.get("feed_id", "未知源")),
                    url=item.get("url", ""),
                    published_at=item.get("published_at", ""),
                    source_kind=item.get("source_kind", "media"),
                )
                for item in candidate.get("items", [])[:preferred_refs]
            ]
            sections.append(
                IndustrySection(
                    key=candidate["bucket_key"],
                    title=candidate["suggested_title"],
                    kind="tag_cluster",
                    summary=self._first_sentence("；".join(ref.title for ref in refs), max_len=100),
                    reference_ids=[ref.id for ref in refs],
                    references=refs,
                )
            )
        return sections

    def _hydrate_references(
        self,
        sections: List[IndustrySection],
        items_by_id: Dict[int, Dict[str, Any]],
        max_refs_per_section: int,
    ) -> List[IndustrySection]:
        hydrated = []
        for section in sections:
            refs = []
            for ref_id in section.reference_ids[:max_refs_per_section]:
                item = items_by_id.get(ref_id)
                if not item:
                    continue
                refs.append(
                    IndustrySectionReference(
                        id=ref_id,
                        title=item.get("title", ""),
                        source_name=item.get("feed_name", item.get("feed_id", "未知源")),
                        url=item.get("url", ""),
                        published_at=item.get("published_at", ""),
                        source_kind=item.get("source_kind", "media"),
                    )
                )
            section.references = refs
            section.reference_ids = [ref.id for ref in refs]
            hydrated.append(section)
        return hydrated

    def _parse_structured_response(
        self,
        response: str,
        group: Dict,
        items_by_id: Dict[int, Dict[str, Any]],
    ) -> IndustryStandaloneReport:
        report = IndustryStandaloneReport(success=False, type=group["TYPE"], display_name=group["DISPLAY_NAME"], raw_response=response)
        strategy = group.get("REPORT", {}).get("SECTION_STRATEGY", {}) or {}
        try:
            start = response.find("{")
            end = response.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("AI 响应中未找到 JSON")
            data = json.loads(response[start:end + 1])
            raw_sections = data.get("sections", []) or []
            sections: List[IndustrySection] = []
            for raw_section in raw_sections[: strategy.get("MAX_SECTIONS", 6)]:
                if not isinstance(raw_section, dict):
                    continue
                title = str(raw_section.get("title", "") or "").strip()
                if not title:
                    continue
                key = str(raw_section.get("key", "") or "").strip() or self._slugify(title)
                kind = str(raw_section.get("kind", "tag_cluster") or "tag_cluster").strip() or "tag_cluster"
                reference_ids = []
                for item in raw_section.get("reference_ids", []) or []:
                    try:
                        reference_ids.append(int(item))
                    except (TypeError, ValueError):
                        continue
                sections.append(
                    IndustrySection(
                        key=key,
                        title=title,
                        kind=kind,
                        summary=str(raw_section.get("summary", "") or "").strip(),
                        reference_ids=reference_ids,
                    )
                )

            sections = self._hydrate_references(
                sections,
                items_by_id,
                strategy.get("MAX_REFS_PER_SECTION", 8),
            )
            sections = [section for section in sections if section.references]
            report.success = True
            report.sections = sections
            report.final_summary = str(data.get("final_summary", "") or "").strip()
            return report
        except Exception as e:
            report.error = f"行业分析解析失败: {e}"
            return report

    def _build_stats(self, items: List[Dict]) -> Dict[str, Any]:
        tags_counter = Counter()
        for item in items:
            tags_counter.update([str(tag).strip() for tag in (item.get("tags") or []) if str(tag).strip()])
        return {
            "item_count": len(items),
            "kol_count": sum(1 for item in items if item.get("source_kind") == "kol"),
            "media_count": sum(1 for item in items if item.get("source_kind") == "media"),
            "top_tags": [key for key, _ in tags_counter.most_common(8)],
        }

    def _build_summary_report(self, group: Dict, standalone_report: IndustryStandaloneReport) -> IndustrySummaryReport:
        summary_cfg = group.get("REPORT", {}).get("SUMMARY", {}) or {}
        max_sections = summary_cfg.get("MAX_SECTIONS", 3)
        summary_sections = [
            IndustrySummarySection(title=section.title, summary=self._first_sentence(section.summary))
            for section in standalone_report.sections[:max_sections]
            if self._first_sentence(section.summary)
        ]
        return IndustrySummaryReport(
            type=standalone_report.type,
            display_name=standalone_report.display_name,
            summary_sections=summary_sections,
            final_summary=self._first_sentence(standalone_report.final_summary),
        )

    def _sync_legacy_fields(self, result: IndustryAnalysisResult) -> None:
        standalone = result.standalone_report
        if not standalone:
            return
        summaries = [section.summary for section in standalone.sections[:6] if section.summary]
        result.summary = standalone.final_summary or (summaries[0] if summaries else "")
        if len(summaries) > 0:
            result.consensus_topics = summaries[0]
        if len(summaries) > 1:
            result.fresh_signals = summaries[1]
        if len(summaries) > 2:
            result.notable_entities = summaries[2]
        divergence = next((section.summary for section in standalone.sections if section.key == self.SPECIAL_SECTION_KEY), "")
        result.kol_media_divergence = divergence
        if len(summaries) > 3:
            result.risk_watch = summaries[3]

    def analyze_group(self, group: Dict, items: List[Dict]) -> IndustryAnalysisResult:
        result = IndustryAnalysisResult(type=group["TYPE"], display_name=group["DISPLAY_NAME"])
        group_name = result.display_name or result.type
        source_kinds = set(group.get("SOURCE_KINDS", ["kol", "media"]))
        freshness = group.get("FRESHNESS", {})
        dedup_enabled = group.get("DEDUP", {}).get("ENABLED", True)
        result.include_in_main_report = group.get("INCLUDE_IN_MAIN_REPORT", True)
        result.standalone_feishu_webhook_url = group.get("NOTIFICATION", {}).get("FEISHU", {}).get("WEBHOOK_URL", "")
        result.send_standalone_report = group.get("NOTIFICATION", {}).get("ENABLED", True) and group.get("NOTIFICATION", {}).get("SEND_STANDALONE_REPORT", False)

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
        if freshness.get("ENABLED", True):
            filtered = self._filter_by_freshness(filtered, freshness.get("HOURS", 24))
        filtered = [item for item in filtered if item.get("source_kind", "media") in source_kinds]
        if dedup_enabled:
            filtered = self._dedup_items(filtered)

        result.raw_count = len(filtered)
        result.kol_count = sum(1 for item in filtered if item.get("source_kind") == "kol")
        result.media_count = sum(1 for item in filtered if item.get("source_kind") == "media")

        if not filtered:
            result.skipped = True
            result.error = "该行业分组暂无可分析 RSS 内容"
            self._log(f"group={group_name} skipped: {result.error}")
            return result

        filtered = self._annotate_items_with_ids(filtered)
        items_by_id = {item["industry_item_id"]: item for item in filtered}
        candidate_sections = self._build_candidate_sections(group, filtered)
        stats = self._build_stats(filtered)

        system_prompt, user_template = load_prompt_template(
            group.get("AI", {}).get("PROMPT_FILE", "industry_analysis_prompt.txt"),
            label="INDUSTRY",
        )
        if not user_template:
            result.error = "行业分析提示词为空"
            self._log(f"group={group_name} failed before AI: {result.error}")
            return result

        system_prompt, user_prompt = self._build_structured_prompt(
            group, filtered, candidate_sections, system_prompt, user_template
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        if self.debug:
            print(f"[Industry][DEBUG] type={group['TYPE']} prompt:\n{user_prompt}")

        try:
            response = self.client.chat(messages)
            standalone = self._parse_structured_response(response, group, items_by_id)
            if not standalone.success or not standalone.sections:
                fallback_sections = self._build_fallback_sections(group, filtered, candidate_sections)
                if fallback_sections:
                    standalone.success = True
                    standalone.sections = fallback_sections
                    standalone.final_summary = standalone.final_summary or self._first_sentence("；".join(section.summary for section in fallback_sections), max_len=120)
                    standalone.error = ""
            standalone.stats = stats
            result.raw_response = response
            result.standalone_report = standalone
            result.summary_report = self._build_summary_report(group, standalone) if standalone.success else None
            result.success = standalone.success
            result.error = standalone.error
            self._sync_legacy_fields(result)
            return result
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

        grouped_items: Dict[str, List[Dict]] = defaultdict(list)
        for item in rss_items or []:
            item_type = str(item.get("feed_type", "") or "").strip()
            if item_type:
                grouped_items[item_type].append(item)

        results: List[IndustryAnalysisResult] = []
        for group in groups:
            results.append(self.analyze_group(group, grouped_items.get(group["TYPE"], [])))
        return results
