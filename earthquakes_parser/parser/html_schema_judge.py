"""Batch schema extraction from links.csv using GPT."""

import glob
import json
import os
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup

from ..search.html_downloader import HTMLDownloader
from .schema_extractor import SchemaExtractor


class HTMLSchemaJudge:
    """Reads links from CSV, fetches HTML, extracts schemas via GPT, saves to data/."""

    DATA_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data")
    )

    def __init__(
        self,
        openai_base_url: str = "http://192.168.8.22:9999/v1",
        openai_api_key: str = "api-key",  # pragma: allowlist secret
        model: str = "gpt-4",
        max_tokens: int = 80000,
        csv_path: Optional[str] = None,
    ):
        """Initialize HTML schema judge with LLM configuration."""
        self.extractor = SchemaExtractor(
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            model=model,
            max_tokens=max_tokens,
        )
        self.downloader = HTMLDownloader(fetch_with="bs4")
        self.csv_path = csv_path or os.path.join(self.DATA_DIR, "links.csv")

    # --- index & loading helpers -----------------------------------------------

    def _next_schema_index(self) -> int:
        """Find next available schema index in data/."""
        idx = 1
        while os.path.exists(os.path.join(self.DATA_DIR, f"schema{idx}.json")):
            idx += 1
        return idx

    @staticmethod
    def _schema_sort_key(filename: str) -> int:
        """Extract numeric index from 'schemaX.json' for sorting."""
        return int(os.path.splitext(filename)[0].replace("schema", ""))

    @staticmethod
    def _load_all_schemas() -> dict[str, list[dict]]:
        """Load every schemaX.json from DATA_DIR, ordered by index.

        Returns:
            {filename: [entries]} e.g. {"schema1.json": [...], "schema2.json": [...]}
        """
        pattern = os.path.join(HTMLSchemaJudge.DATA_DIR, "schema*.json")
        result: dict[str, list[dict]] = {}
        for path in glob.glob(pattern):
            name = os.path.basename(path)
            stem = os.path.splitext(name)[0].replace("schema", "")
            if stem.isdigit():
                with open(path, "r", encoding="utf-8") as f:
                    result[name] = json.load(f)
        return dict(
            sorted(
                result.items(),
                key=lambda item: HTMLSchemaJudge._schema_sort_key(item[0]),
            )
        )

    @staticmethod
    def _load_model_schemas() -> dict[str, list[dict]]:
        """Load every schema-<model>.json from DATA_DIR.

        Returns:
            {model_name: [entries]}
            e.g. {"gemma": [...], "gpt_oss": [...], "llama": [...]}
        """
        pattern = os.path.join(HTMLSchemaJudge.DATA_DIR, "schema-*.json")
        result: dict[str, list[dict]] = {}
        for path in glob.glob(pattern):
            name = os.path.basename(path)
            # Extract model name: schema-gemma.json -> gemma
            if name.startswith("schema-") and name.endswith(".json"):
                model_name = name[7:-5]  # Remove "schema-" prefix and ".json" suffix
                with open(path, "r", encoding="utf-8") as f:
                    result[model_name] = json.load(f)
        return dict(sorted(result.items()))

    # --- single-URL helpers ----------------------------------------------------

    def _load_links(self) -> pd.DataFrame:
        """Load links DataFrame from CSV."""
        return pd.read_csv(self.csv_path)

    @staticmethod
    def _get_domain(url: str) -> str:
        """Extract domain from URL."""
        return urlparse(url).netloc

    @staticmethod
    def _get_title(html: str) -> str:
        """Extract page title from <title> tag."""
        soup = BeautifulSoup(html, "html.parser")
        tag = soup.find("title")
        return tag.get_text(strip=True) if tag else "Untitled"

    def _process_url(self, url: str) -> Optional[dict]:
        """Fetch HTML and extract schema for a single URL."""
        try:
            html = self.downloader.fetch_html(url)
            title = self._get_title(html)
            domain = self._get_domain(url)

            schema = self.extractor.extract_schema(html, title, domain)
            if schema is None:
                return None

            return {
                "url": url,
                "domain": domain,
                "title": title,
                "schema": schema.to_dict(),
                "is_valid": schema.is_valid,
            }
        except Exception as e:
            print(f"Error processing {url}: {e}")
            return None

    # --- extraction run --------------------------------------------------------

    def run(self, compare_after: bool = False) -> str:
        """Process all links from CSV, save schemas to data/schemaX.json.

        Args:
            compare_after: run compare_all() automatically after writing,
                           only when 2+ schema files already exist.

        Returns:
            Path to the saved schema file.
        """
        links_df = self._load_links()
        idx = self._next_schema_index()
        output_path = os.path.join(self.DATA_DIR, f"schema{idx}.json")

        results = []
        for i, (_, row) in enumerate(links_df.iterrows()):
            url = row["link"]
            print(f"\n[{i + 1}/{len(links_df)}] {url}")

            result = self._process_url(url)
            if result:
                results.append(result)
                print(f"  -> {result['domain']}")
            else:
                results.append({"url": url, "error": "extraction failed"})
                print("  -> skipped")

        os.makedirs(self.DATA_DIR, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        success = sum(1 for r in results if "schema" in r)
        print(f"\n[done] {success}/{len(results)} schemas saved -> {output_path}")

        if compare_after and idx >= 2:
            print("\n" + "=" * 60)
            self.compare_all()

        return output_path

    # --- GPT judging -----------------------------------------------------------

    def _truncate_for_judge(self, html: str) -> str:
        """Truncate HTML to leave room for the judge prompt overhead."""
        budget = self.extractor.max_tokens - 2000  # reserve for prompt + schema text
        token_count = self.extractor.count_tokens(html)
        if token_count > budget:
            max_chars = int(len(html) * (budget / token_count * 0.9))
            return html[:max_chars]
        return html

    def _build_judge_prompt(
        self,
        html: str,
        domain: str,
        schemas_by_run: dict[str, dict],
        consistent: bool,
    ) -> str:
        """Build GPT prompt for judging / validating schemas of one domain.

        consistent=True  →  all runs agree, ask GPT to validate correctness.
        consistent=False →  runs disagree, ask GPT to pick the best one.
        """
        if consistent:
            schema = next(iter(schemas_by_run.values()))
            return (
                f'You are given the HTML of a page from the domain "{domain}".\n'
                f"Validate whether the CSS selectors below correctly point to the\n"
                f"main content and the publication date on this page.\n\n"
                f"HTML:\n{html}\n\n"
                f"Schema:\n"
                f"  main_text_selectors = {schema.get('main_text_selectors', [])}\n"
                f"  date_selector       = {schema.get('date_selector')}\n\n"
                "Return ONLY a JSON object wrapped in ```json``` blocks:\n"
                "```json\n"
                '{"valid": <true|false>, "confidence": <0.0-1.0>, '
                '"reason": "<short explanation>"}\n'
                "```\n"
            )

        schemas_lines = "\n".join(
            f"  {run}: main_text={s.get('main_text_selectors', [])}, "
            f"date={s.get('date_selector')}"
            for run, s in sorted(
                schemas_by_run.items(),
                key=lambda x: HTMLSchemaJudge._schema_sort_key(x[0]),
            )
        )
        return (
            f'You are given the HTML of a page from the domain "{domain}".\n'
            f"CSS selectors were extracted across multiple runs and they differ.\n"
            f"Pick the run that produced the most accurate selectors.\n\n"
            f"HTML:\n{html}\n\n"
            f"Schemas by run:\n{schemas_lines}\n\n"
            "Return ONLY a JSON object wrapped in ```json``` blocks:\n"
            "```json\n"
            '{"best_run": "<run filename>", "confidence": <0.0-1.0>, '
            '"reason": "<short explanation>"}\n'
            "```\n"
        )

    def _judge_domain(
        self,
        url: str,
        domain: str,
        schemas_by_run: dict[str, dict],
        consistent: bool,
    ) -> Optional[dict]:
        """Fetch HTML for *url* and call GPT to judge schemas for *domain*."""
        try:
            print(f"  [judge] {domain}  ({url})")
            html = self.downloader.fetch_html(url)
            html = self._truncate_for_judge(html)
            prompt = self._build_judge_prompt(html, domain, schemas_by_run, consistent)

            response = self.extractor.client.chat.completions.create(
                model=self.extractor.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at validating CSS selectors "
                            "against HTML page structure."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            result = self.extractor._extract_json(response.choices[0].message.content)
            if result:
                result["domain"] = domain
                result["url"] = url
                result["consistent"] = consistent
            return result
        except Exception as e:
            print(f"  [judge] error for {domain}: {e}")
            return None

    # --- model evaluation ------------------------------------------------------

    def _evaluate_selectors_on_html(
        self,
        html: str,
        schema: dict,
    ) -> dict:
        """Test selectors on real HTML and calculate extraction accuracy metrics.

        Returns:
            {
                "main_text_found": bool,
                "main_text_length": int,
                "date_found": bool,
                "extraction_accuracy_score": float (0.0-1.0)
            }
        """
        soup = BeautifulSoup(html, "html.parser")

        # Test main_text_selectors
        main_text_selectors = schema.get("main_text_selectors", [])
        main_text_parts = []
        for selector in main_text_selectors:
            try:
                elements = soup.select(selector)
                for elem in elements:
                    text = elem.get_text(strip=True)
                    if text:
                        main_text_parts.append(text)
            except Exception:
                # Invalid selector, skip
                continue  # nosec B112

        main_text = " ".join(main_text_parts)
        main_text_found = len(main_text) > 50  # At least 50 chars for valid content

        # Test date_selector
        date_selector = schema.get("date_selector")
        date_found = False
        if date_selector:
            try:
                date_elem = soup.select_one(date_selector)
                if date_elem:
                    date_text = (
                        date_elem.get_text(strip=True)
                        if date_elem.name != "meta"
                        else date_elem.get("content", "")
                    )
                    date_found = len(date_text) > 0
            except Exception:
                # Invalid date selector, skip
                pass  # nosec B110

        # Calculate extraction accuracy score
        extraction_accuracy_score = 0.0
        if main_text_found:
            extraction_accuracy_score += 0.7  # 70% weight for main text
        if date_found:
            extraction_accuracy_score += 0.3  # 30% weight for date

        return {
            "main_text_found": main_text_found,
            "main_text_length": len(main_text),
            "date_found": date_found,
            "extraction_accuracy_score": round(extraction_accuracy_score, 2),
        }

    def _judge_model_selectors(
        self,
        html: str,
        url: str,
        domain: str,
        schema: dict,
    ) -> Optional[dict]:
        """Use GPT to judge the quality of a model's selectors on a 0-10 scale.

        Returns:
            {
                "score": float (0.0-10.0),
                "confidence": float (0.0-1.0),
                "reason": {
                    "description": str,
                    "evaluation_criteria": dict,
                    "error_breakdown": list
                },
                "main_text_quality": float,
                "date_quality": float,
                "issues": list,
                "suggestions": str
            }
        """
        try:
            html = self._truncate_for_judge(html)

            prompt = (
                "You are an expert CSS selector evaluator for web scraping.\n"
                f'Given HTML from domain "{domain}", evaluate the provided '
                "selectors on a scale of 0-10.\n\n"
                "EVALUATION CRITERIA:\n\n"
                "1. MAIN TEXT SELECTORS (70% weight):\n"
                "   - Do they target the actual article body/content?\n"
                "   - Do they capture ALL relevant paragraphs, headings, quotes?\n"
                "   - Do they AVOID navigation, sidebars, ads, footers, comments?\n"
                "   - Are they specific enough but not overly fragile?\n"
                "   - Do they exist in the DOM (check if elements match)?\n\n"
                "2. DATE SELECTOR (30% weight):\n"
                "   - Does it point to the publication/update date?\n"
                "   - Is the element present in the HTML?\n"
                "   - Does it avoid capturing wrong timestamps?\n\n"
                "3. QUALITY FACTORS:\n"
                "   - Precision: How much irrelevant content is captured?\n"
                "   - Recall: How much relevant content is missed?\n"
                "   - Robustness: Will selectors break with minor HTML changes?\n"
                "   - Reusability: Are selectors too page-specific "
                "(hardcoded IDs)?\n\n"
                "ERROR CATEGORIES (categorize each issue from 'issues' field):\n"
                "   - Missing Content: Important content not captured "
                "(paragraphs, quotes, lists, etc)\n"
                "   - Excessive Noise: Unwanted content captured "
                "(ads, navigation, sidebars, footers)\n"
                "   - Wrong Element: Selector targets wrong element "
                "or doesn't exist in DOM\n"
                "   - Fragility: Selector is too brittle "
                "(nth-child, long class chains, page-specific IDs)\n\n"
                "SEVERITY LEVELS for penalty calculation:\n"
                "   - critical: -2.0 points (missing main content, "
                "wrong date, selector doesn't exist)\n"
                "   - medium: -1.0 point (noise, minor missing content, "
                "moderately fragile)\n"
                "   - minor: -0.5 points (small fragility issues, edge cases)\n\n"
                "SCORING GUIDE:\n"
                f"  9-10: Excellent - precise, complete, robust selectors\n"
                f"  7-8:  Good - captures most content with minor noise\n"
                f"  5-6:  Fair - works but has significant issues\n"
                f"  3-4:  Poor - misses content or captures too much noise\n"
                f"  1-2:  Failed - selectors don't exist or completely wrong\n\n"
                f"HTML:\n{html}\n\n"
                f"SELECTORS TO EVALUATE:\n"
                f"  main_text_selectors = {schema.get('main_text_selectors', [])}\n"
                f"  date_selector       = {schema.get('date_selector')}\n\n"
                f"INSTRUCTIONS:\n"
                f"1. Test each selector against the HTML mentally\n"
                f"2. Identify what content would be captured\n"
                f"3. Check for false positives (unwanted content)\n"
                f"4. Check for false negatives (missed content)\n"
                f"5. Categorize each issue into one of the error categories\n"
                f"6. Assign severity level to each error\n"
                f"7. Calculate penalty scores\n\n"
                "Return ONLY a JSON object wrapped in ```json``` blocks:\n"
                "```json\n"
                "{\n"
                '  "score": <0.0-10.0>,\n'
                '  "confidence": <0.0-1.0>,\n'
                '  "reason": {\n'
                '    "description": "<overall explanation with specific examples>",\n'
                '    "evaluation_criteria": {\n'
                '      "precision": {"score": <0-10>, "max_score": 10, '
                '"issues_count": <int>},\n'
                '      "recall": {"score": <0-10>, "max_score": 10, '
                '"issues_count": <int>},\n'
                '      "robustness": {"score": <0-10>, "max_score": 10, '
                '"issues_count": <int>}\n'
                "    },\n"
                '    "error_breakdown": [\n'
                "      {\n"
                '        "category": '
                '"<Missing Content|Excessive Noise|Wrong Element|Fragility>",\n'
                '        "severity": "<critical|medium|minor>",\n'
                '        "count": <number of issues in this category>,\n'
                '        "penalty": <total penalty for this category>,\n'
                '        "examples": ["<issue example 1>", "<issue example 2>"]\n'
                "      }\n"
                "    ]\n"
                "  },\n"
                '  "main_text_quality": <0.0-10.0>,\n'
                '  "date_quality": <0.0-10.0>,\n'
                '  "issues": ["<specific issue 1>", "<specific issue 2>"],\n'
                '  "suggestions": "<brief improvement suggestion>"\n'
                "}\n"
                "```\n"
            )

            response = self.extractor.client.chat.completions.create(
                model=self.extractor.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert at evaluating CSS selectors "
                            "for web scraping quality and accuracy."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )

            result = self.extractor._extract_json(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"    [error] GPT judge failed: {e}")
            return None

    def evaluate_models(self) -> dict:
        """Evaluate all models in schema-<model>.json files.

        For each model:
        - Calculate success rate (40% weight)
        - Test selectors on real HTML and calculate extraction accuracy (30% weight)
        - Use GPT to judge selector quality (30% weight)
        - Compute final score out of 100

        Returns:
            {
                "models": {
                    "model_name": {
                        "success_rate": float,
                        "extraction_accuracy_avg": float,
                        "gpt_quality_avg": float,
                        "final_score": float (0-100),
                        "details": [...]
                    }
                }
            }
        """
        model_schemas = self._load_model_schemas()

        if not model_schemas:
            print("[evaluate] No schema-*.json files found in data/")
            return {}

        links_df = self._load_links()
        results = {}

        print(f"\n{'=' * 70}")
        print(
            f"  Model Evaluation: {len(model_schemas)} models x {len(links_df)} links"
        )
        print(f"{'=' * 70}\n")

        for model_name, entries in model_schemas.items():
            print(f"[{model_name}] Evaluating...")

            total_links = len(links_df)
            successful_extractions = [e for e in entries if "schema" in e]
            success_count = len(successful_extractions)
            success_rate = success_count / total_links if total_links > 0 else 0.0

            # Evaluate each successful extraction
            extraction_accuracy_scores = []
            gpt_scores = []
            details = []

            for i, entry in enumerate(successful_extractions[:20]):  # Limit to 20 links
                url = entry["url"]
                schema = entry["schema"]
                domain = entry.get("domain", "unknown")

                print(f"  [{i+1}/{min(success_count, 20)}] {domain}")

                try:
                    # Fetch HTML
                    html = self.downloader.fetch_html(url)

                    # Test selectors on real HTML
                    extraction_accuracy_result = self._evaluate_selectors_on_html(
                        html, schema
                    )
                    extraction_accuracy_scores.append(
                        extraction_accuracy_result["extraction_accuracy_score"]
                    )

                    # GPT judge
                    gpt_result = self._judge_model_selectors(html, url, domain, schema)
                    if gpt_result and "score" in gpt_result:
                        normalized_score = (
                            gpt_result["score"] / 10.0
                        )  # Normalize to 0-1
                        gpt_scores.append(normalized_score)
                    else:
                        gpt_scores.append(0.0)

                    details.append(
                        {
                            "url": url,
                            "domain": domain,
                            "extraction_accuracy": extraction_accuracy_result,
                            "gpt_judgment": gpt_result,
                        }
                    )

                except Exception as e:
                    print(f"    [error] {e}")
                    details.append(
                        {
                            "url": url,
                            "domain": domain,
                            "error": str(e),
                        }
                    )

            # Calculate averages
            extraction_accuracy_avg = (
                sum(extraction_accuracy_scores) / len(extraction_accuracy_scores)
                if extraction_accuracy_scores
                else 0.0
            )
            gpt_quality_avg = sum(gpt_scores) / len(gpt_scores) if gpt_scores else 0.0

            # Final score: 40% success rate + 30% extraction accuracy + 30% GPT quality
            final_score = (
                success_rate * 40 + extraction_accuracy_avg * 30 + gpt_quality_avg * 30
            )

            results[model_name] = {
                "total_links": total_links,
                "success_count": success_count,
                "success_rate": round(success_rate, 3),
                "extraction_accuracy_avg": round(extraction_accuracy_avg, 3),
                "gpt_quality_avg": round(gpt_quality_avg, 3),
                "final_score": round(final_score, 2),
                "details": details,
            }

            print(
                f"  -> Success: {success_count}/{total_links} ({success_rate*100:.1f}%)"
            )
            print(f"  -> Extraction Accuracy: {extraction_accuracy_avg:.2f}")
            print(f"  -> GPT Quality: {gpt_quality_avg:.2f}")
            print(f"  -> Final Score: {final_score:.2f}/100\n")

        # Save results
        evaluation_path = os.path.join(self.DATA_DIR, "model_evaluation.json")
        with open(evaluation_path, "w", encoding="utf-8") as f:
            json.dump({"models": results}, f, ensure_ascii=False, indent=2)

        print(f"[evaluate] Results saved -> {evaluation_path}\n")

        # Print summary
        _print_model_evaluation(results)

        return {"models": results}

    # --- comparison ------------------------------------------------------------

    def compare_all(self) -> dict:
        """Compare all schemaX.json files across runs, with GPT accuracy judging.

        For every domain that appears in 2+ runs:
          consistent selectors  ->  GPT validates correctness
          differing selectors   ->  GPT picks the best run

        Prints a report and saves metrics to data/comparison.json.

        Returns:
            Metrics dict.  Empty dict when fewer than 2 files exist.
        """
        all_schemas = self._load_all_schemas()

        if len(all_schemas) < 2:
            print("[compare] need at least 2 schema files in data/.")
            return {}

        # ---- per-run stats ----------------------------------------------------
        per_run: list[dict] = []
        domain_runs: dict[str, dict[str, dict]] = {}
        domain_urls: dict[str, str] = {}  # first URL seen per domain

        for run_name, entries in all_schemas.items():
            total = len(entries)
            successful = [e for e in entries if "schema" in e]
            success_count = len(successful)
            valid_count = sum(1 for e in successful if e.get("is_valid", False))

            per_run.append(
                {
                    "file": run_name,
                    "total": total,
                    "success": success_count,
                    "success_rate": round(success_count / total, 2) if total else 0.0,
                    "valid": valid_count,
                    "valid_rate": round(valid_count / total, 2) if total else 0.0,
                }
            )

            for entry in successful:
                domain = entry["domain"]
                domain_runs.setdefault(domain, {})[run_name] = entry["schema"]
                domain_urls.setdefault(domain, entry["url"])

        # ---- stability: compare selectors for domains seen in 2+ runs ---------
        shared_domains = {d: runs for d, runs in domain_runs.items() if len(runs) >= 2}

        stability: list[dict] = []
        for domain in sorted(shared_domains):
            runs = shared_domains[domain]
            run_names = sorted(runs.keys(), key=self._schema_sort_key)

            snapshots = [
                (
                    tuple(sorted(runs[r].get("main_text_selectors", []))),
                    runs[r].get("date_selector"),
                )
                for r in run_names
            ]
            consistent = all(s == snapshots[0] for s in snapshots)

            stability.append(
                {
                    "domain": domain,
                    "appearances": len(run_names),
                    "runs": run_names,
                    "consistent": consistent,
                    "selectors": {
                        r: {
                            "main_text": runs[r].get("main_text_selectors", []),
                            "date": runs[r].get("date_selector"),
                        }
                        for r in run_names
                    },
                }
            )

        # ---- GPT judging shared domains ---------------------------------------
        print("\n[compare] GPT judging shared domains...")
        judgments: list[dict] = []
        for s in stability:
            domain = s["domain"]
            judgment = self._judge_domain(
                url=domain_urls[domain],
                domain=domain,
                schemas_by_run=shared_domains[domain],
                consistent=s["consistent"],
            )
            judgments.append(
                judgment
                if judgment
                else {
                    "domain": domain,
                    "url": domain_urls[domain],
                    "error": "judgment failed",
                }
            )

        # ---- assemble ---------------------------------------------------------
        metrics: dict = {
            "runs": len(all_schemas),
            "per_run": per_run,
            "shared_domains": sorted(shared_domains.keys()),
            "unique_domains": sorted(
                d for d, runs in domain_runs.items() if len(runs) == 1
            ),
            "stability": stability,
            "judgments": judgments,
        }

        _print_comparison(metrics)

        # ---- persist ----------------------------------------------------------
        comparison_path = os.path.join(self.DATA_DIR, "comparison.json")
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\n[compare] saved -> {comparison_path}")

        return metrics


# ---------------------------------------------------------------------------
# pretty-print helper (module-level, not part of the public API)
# ---------------------------------------------------------------------------


def _print_model_evaluation(results: dict) -> None:
    """Print a human-readable model evaluation report to stdout."""
    print("\n" + "=" * 80)
    print("  Model Evaluation Summary")
    print("=" * 80 + "\n")

    # Sort models by final score (descending)
    sorted_models = sorted(
        results.items(), key=lambda x: x[1]["final_score"], reverse=True
    )

    # Main summary table
    header = (
        f"  {'Model':<15} {'Success':>8} "
        f"{'Extract Acc':>11} {'GPT Qual':>10} {'Score':>8}"
    )
    print(header)
    separator = f"  {'-' * 15} {'-' * 8} {'-' * 11} {'-' * 10} {'-' * 8}"
    print(separator)

    for model_name, metrics in sorted_models:
        success_rate = metrics["success_rate"] * 100
        extraction_accuracy = metrics["extraction_accuracy_avg"] * 100
        gpt_quality = metrics["gpt_quality_avg"] * 100
        final_score = metrics["final_score"]

        print(
            f"  {model_name:<15} "
            f"{success_rate:>7.1f}% "
            f"{extraction_accuracy:>10.1f}% "
            f"{gpt_quality:>9.1f}% "
            f"{final_score:>7.1f}"
        )

    # Winner
    if sorted_models:
        winner = sorted_models[0]
        print(
            f"\n  🏆 Winner: {winner[0]} with {winner[1]['final_score']:.2f}/100 points"
        )

    # Detailed breakdown for top model
    if sorted_models:
        print("\n" + "=" * 80)
        print(f"  Top Model Detailed Analysis: {sorted_models[0][0]}")
        print("=" * 80 + "\n")

        top_model = sorted_models[0][1]

        # Count issues
        total_issues = 0
        common_issues: dict[str, int] = {}

        for detail in top_model.get("details", []):
            if "gpt_judgment" in detail and detail["gpt_judgment"]:
                judgment = detail["gpt_judgment"]

                # Count issues
                if "issues" in judgment:
                    total_issues += len(judgment["issues"])
                    for issue in judgment["issues"]:
                        common_issues[issue] = common_issues.get(issue, 0) + 1

        print(f"  Total Sites Evaluated: {top_model['success_count']}")
        print("  Average Scores:")

        # Calculate average main text quality
        main_text_scores = [
            d.get("gpt_judgment", {}).get("main_text_quality", 0)
            for d in top_model.get("details", [])
            if d.get("gpt_judgment")
        ]
        avg_main_text = sum(main_text_scores) / max(len(main_text_scores), 1)
        print(f"    - Main Text Quality: {avg_main_text:.1f}/10")

        # Calculate average date quality
        date_scores = [
            d.get("gpt_judgment", {}).get("date_quality", 0)
            for d in top_model.get("details", [])
            if d.get("gpt_judgment")
        ]
        avg_date = sum(date_scores) / max(len(date_scores), 1)
        print(f"    - Date Quality: {avg_date:.1f}/10")

        print(f"\n  Issues Found: {total_issues}")

        if common_issues:
            print("\n  Most Common Issues:")
            for issue, count in sorted(
                common_issues.items(), key=lambda x: x[1], reverse=True
            )[:3]:
                print(f"    - {issue} ({count} times)")

    print(f"\n{'=' * 80}\n")


def _print_comparison(metrics: dict) -> None:
    """Print a human-readable comparison report to stdout."""
    print("\n" + "=" * 62)
    print(f"  Schema comparison   ({metrics['runs']} runs)")
    print("=" * 62)

    # --- per-run table -----------------------------------------------------
    header = (
        f"\n  {'File':<16} {'Total':>5} "
        f"{'OK':>4} {'OK%':>5} {'Valid':>5} {'Vld%':>5}"
    )
    print(header)
    separator = f"  {'-' * 16} {'-' * 5} {'-' * 4} {'-' * 5} {'-' * 5} {'-' * 5}"
    print(separator)
    for r in metrics["per_run"]:
        print(
            f"  {r['file']:<16} "
            f"{r['total']:>5} "
            f"{r['success']:>4} "
            f"{r['success_rate'] * 100:>4.0f}% "
            f"{r['valid']:>5} "
            f"{r['valid_rate'] * 100:>4.0f}%"
        )

    # --- domain summary ----------------------------------------------------
    shared = metrics["shared_domains"]
    unique = metrics["unique_domains"]
    total_domains = len(shared) + len(unique)
    domains_info = (
        f"\n  Domains:  shared={len(shared)}  "
        f"unique={len(unique)}  total={total_domains}"
    )
    print(domains_info)

    # --- stability table ---------------------------------------------------
    if metrics["stability"]:
        stable_count = sum(1 for s in metrics["stability"] if s["consistent"])
        total_shared = len(metrics["stability"])
        stability_msg = (
            f"  Stability: {stable_count}/{total_shared} "
            "shared domains have identical selectors\n"
        )
        print(stability_msg)

        header = f"  {'Domain':<38} {'Runs':>4} {'Stable':>6}"
        print(header)
        separator = f"  {'-' * 38} {'-' * 4} {'-' * 6}"
        print(separator)
        for s in metrics["stability"]:
            label = "YES" if s["consistent"] else "NO"
            print(f"  {s['domain']:<38} {s['appearances']:>4} {label:>6}")

            if not s["consistent"]:
                for run_name, sels in s["selectors"].items():
                    print(f"      {run_name:<18} main_text = {sels['main_text']}")
                    print(f"      {'':18} date      = {sels['date']}")

    # --- GPT judgments -----------------------------------------------------
    if metrics.get("judgments"):
        print("\n  GPT Judgments")
        separator = f"  {'-' * 60}"
        print(separator)
        for j in metrics["judgments"]:
            if "error" in j:
                print(f"  {j['domain']:<38} [error]")
                continue

            if "best_run" in j:
                # unstable domain — GPT picked a winner
                print(
                    f"  {j['domain']:<38} "
                    f"best={j['best_run']:<16} "
                    f"conf={j.get('confidence', '?')}"
                )
            elif "valid" in j:
                # stable domain — GPT validated correctness
                status = "OK" if j["valid"] else "INVALID"
                print(
                    f"  {j['domain']:<38} "
                    f"{status:<8} "
                    f"conf={j.get('confidence', '?')}"
                )

            if j.get("reason"):
                print(f"      -> {j['reason']}")

    print(f"\n{'=' * 62}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HTMLSchemaJudge — extract & compare page schemas"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="extract schemas from links.csv -> data/schemaX.json",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare all existing schema files (GPT judges accuracy)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate all models in schema-<model>.json files with scoring out of 100",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="path to CSV with links (default: data/links.csv)",
    )
    args = parser.parse_args()

    judge = HTMLSchemaJudge(csv_path=args.csv)

    if args.evaluate:
        judge.evaluate_models()
    elif args.compare and not args.run:
        judge.compare_all()
    elif args.run:
        judge.run(compare_after=args.compare)
    else:
        judge.run()
