import argparse
import asyncio
import itertools
import json
import logging
import statistics
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Literal

import dotenv

# Runtime helpers (env validation, banners, dependency-warning suppression).
from bot_helpers import (
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    DateQuestion,
    DatePercentile,
    Percentile,
    ConditionalQuestion,
    ConditionalPrediction,
    PredictionTypes,
    PredictionAffirmed,
    BinaryPrediction,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Winner-pattern upgrades (2026-11-16)
# Ported from Panshul42 (1st place, Q2 2025) and nostreambot (15th, Summer
# 2026). Full comparison: see the operator's winner-patterns.md notes.
# Deliberately NOT copied: Panshul42's cross-wired forecaster contexts (each
# forecaster here sees ONLY its own stage-1 output), its dead proxy path and
# NUM_RUNS_PER_QUESTION, nostreambot's stacker (88-question test: no better
# than median), and the deprecated community benchmark (Metaculus removed the
# `aggregations` field from the list API).
# ---------------------------------------------------------------------------

# Three forecasters, one per vendor, all routed through litellm + OpenRouter
# (OPENROUTER_API_KEY). Slugs verified against the live OpenRouter model list
# on 2026-11-16. If a slug ever 404s, that forecaster is skipped for the run
# by the per-forecaster isolation in _run_forecast_on_* (the framework's
# required_successful_predictions=0.5 tolerates a missing member); replace
# the slug with the current OpenRouter listing.
FORECASTER_MODELS = {
    "forecaster_a": "openrouter/openai/gpt-6.1-sol",
    "forecaster_b": "openrouter/anthropic/claude-opus-5.5",
    "forecaster_c": "openrouter/google/gemini-4-argon",
}

# Skip a question that cannot be published before it closes.
MIN_MINUTES_TO_CLOSE = 15

# Gap-fill pass: at most this many missing facts get a paid search each.
GAP_FILL_MAX_GAPS = 4

# Numeric/date elicitation: 13 percentiles (nostreambot) instead of 6.
NUMERIC_PERCENTILES = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)

# Measured-failure prompt rules (nostreambot, docs/prompts.md). Each rule
# corrects a failure the bot's own history measured; state once, share
# verbatim across the forecaster prompts.
MEASURED_FAILURE_RULES = """
Reasoning rules (each corrects a failure this kind of bot has measured):
- Separate facts from opinions. Only weight opinions strongly when they come from identifiable experts or credentialed entities.
- Weight factual claims by proximity to the primary record. Claims in the research arrive tagged by source tier where it was clear: [A: ...] official or primary record (including the question's own resolution source), [B: ...] wire services and papers of record, [C: ...] aggregators, advocacy or single-outlet reports (use their cited facts, not their framing), [D: ...] anonymous, social or untraceable (suggestive only).
- `[unverified attribution]` marks a claim whose tag named no outlet: treat it as untiered, unattributed evidence rather than as a named outlet's authority, and not as a low tier either.
- Weigh motivation, not just authority: discount claims that serve the speaker's interest (hype, marketing, sponsor optimism). Treat a statement AGAINST the speaker's interest as strong evidence.
- Primary-record override: an interested party's own filing is still tier-A for the facts it formally attests, even though the party is biased.
- Implausibility check: a figure that is internally implausible or off by ~an order of magnitude versus corroborating sources is likely a transcription or translation error — flag it, don't anchor on it.
- Read a null search result as a null search result. "No record found", "no authoritative source located", or "could not confirm" licenses only "we could not find evidence of X" — it does NOT establish that X does not exist or did not happen. Weight the absence by how well the topic is covered: silence from a comprehensive, well-indexed source the domain reliably reports through is weak-to-moderate evidence; silence from general web search on a poorly-covered topic is nearly no evidence at all.
- A target date the responsible actor has not bound itself to — no statute, no contract, no published schedule it has a measured record of meeting — is evidence that a target EXISTS, not that it will hold. Price the probability that the target lands inside the question window as its own number, derived from that actor's record of slips and scrubs for this kind of event; an announcement, plan, tracker page or partner page does not raise it. Where a binding clock exists, compute the date from it and say which clock. (Announced-but-unbound dates are this bot family's most consistent miss: forecasts averaged 44% on events that happened 8% of the time.)
- If your own analysis names a reason the historical cadence has been discharged (its driver was met, the deadline passed, the rule changed), that cadence is a bound on your estimate, not its center; state the post-change estimate and what it rests on.
- Rates apply to the exposure that REMAINS: estimate the rate over the longest window the evidence supports, then apply it from now until the deadline, treating the elapsed event-free part of the window as observed.
- For questions asking how many events of a kind occur in a period, the admissible outside view is the pooled realized rate over the longest comparable history. A schedule of currently known candidates is evidence about the pipeline; it does not replace the rate.
- Prediction-market snapshot (when provided below): a market on the same quantity at another date/threshold/source is extrapolated from, not vaguely discounted. Weigh a matched market by its signal: deep is close to the best estimate available; strong is serious evidence; decent is a modest input; thin is weak evidence, never an anchor. Below deep, widen around the market's implied value rather than transplanting its price, and read its bid-ask range: a wide range means the price says little. Never treat one outcome's price as an equality constraint that fixes a tail.
- You will be judged on the accuracy and calibration of your forecast under Metaculus' spot peer log score, a proper score: your honest forecast is the best submission whatever other forecasters say.
"""

# Research-side source-tier instruction (nostreambot). Tags must name the
# specific outlet or publisher; a category ("official", "aggregator") is not
# a name. Tag only when the outlet can be named and the tier is clear.
SOURCE_TIER_TAG_INSTRUCTION = """
SOURCE TIER TAGS: annotate each factual claim inline with its source tier AND the specific outlet or publisher it came from, e.g. "[A: BLS]", "[B: Reuters]", "[C: Wikipedia]", "[D: Reddit]".
Tiers: (A) official / primary — government statistics, regulatory filings, court records, central-bank releases, and the question's own named resolution source; (B) wire services and papers of record carrying named-sourced facts; (C) aggregators, advocacy or partisan outlets, and translated or single-outlet reports; (D) anonymous, social, rumor, or untraceable AI-generated summaries.
Tag only when you can name the outlet and the tier is reasonably clear; otherwise leave the claim untagged. NEVER discard a fact because its tier is low: low-tier facts stay in, tagged.
"""

_STOPWORDS = frozenset(
    "a an the and or of to in on for by with will would should could is are was were be been it its this that what when where who whom how many much more most other some any no not than then from at as per via".split()
)


def _market_keywords(text: str) -> str:
    """Short keyword query for the public prediction-market search APIs."""
    words = []
    for raw in text.replace("?", " ").replace(",", " ").split():
        word = "".join(ch for ch in raw if ch.isalnum()).lower()
        if len(word) > 2 and word not in _STOPWORDS and word not in words:
            words.append(word)
        if len(words) == 6:
            break
    return " ".join(words)



class FallTemplateBot2026(ForecastBot):
    """
    This is the template bot for the Fall 2026 FutureEval Bot Tournament.
    This is a copy of what is used by Metaculus to run the Metac Bots in our benchmark, provided as a template for new bot makers.
    This template is given as-is, and is use-at-your-own-risk.
    We have covered most test cases in forecasting-tools it may be worth double checking key components locally.
    So far our track record has been 1 mentionable bug per season (affecting forecasts for 1-2% of total questions)

    This bot is identical to SummerTemplateBot2026.

    The main entry point of this bot is `bot.forecast_on_tournament(tournament_id)` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Alternatively, you can use the MetaculusClient to make a custom filter of questions to forecast on
    and forecast them with `bot.forecast_questions(questions)`

    Only the research and forecast functions need to be implemented in ForecastBot subclasses,
    though you may want to override other ForecastBot functions.
    In this example, you can change the prompts to be whatever you want since,
    structure_output uses an LLM to intelligently reformat the output into the needed structure.

    By default (i.e. 'tournament' mode), when you run this script, it will forecast on any open questions in the
    primary bot tournament and MiniBench. If you want to forecast on only one or the other, you can remove one
    of them from the 'tournament' mode code at the bottom of the file.

    You can experiment with what models work best with your bot by using the `llms` parameter when initializing the bot.
    You can initialize the bot with any number of models. For example,
    ```python
    my_bot = MyBot(
        ...
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o", # "anthropic/claude-sonnet-4-20250514", etc (see docs for litellm)
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
            "researcher": "asknews/news-summaries",
            "parser": "openai/gpt-4o-mini",
        },
    )
    ```

    Then you can access the model in custom functions like this:
    ```python
    research_strategy = self.get_llm("researcher", "model_name")
    if research_strategy == "asknews/news-summaries":
        ...
    # OR
    summarizer = await self.get_llm("summarizer", "llm").invoke(prompt)
    # OR
    reasoning = await self.get_llm("default", "llm").invoke(prompt)
    ```

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```python
    from forecasting_tools import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = (
        1  # Set this to whatever works for your search-provider/ai-model rate limits
    )
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)
    _structure_output_validation_samples = 2

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Round-robin over the three forecasters. _next_forecaster_llm is
        # called synchronously at the top of each _run_forecast_on_* before
        # any await, so the advance is atomic under asyncio's
        # single-threaded loop even though the framework runs the
        # predictions concurrently.
        self._forecaster_cycle = itertools.cycle(FORECASTER_MODELS.keys())

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        defaults = super()._llm_config_defaults()
        for key, model in FORECASTER_MODELS.items():
            defaults[key] = GeneralLlm(
                model=model, temperature=0.3, timeout=160
            )
        return defaults

    def _next_forecaster_llm(self) -> GeneralLlm:
        return self.get_llm(next(self._forecaster_cycle), "llm")

    @staticmethod
    def _ensure_time_to_publish(question: MetaculusQuestion) -> None:
        # Time budget + publish gate (nostreambot time_budget.py /
        # publish_gate.py). A question closing within MIN_MINUTES_TO_CLOSE
        # is skipped before any spend; the same check runs again right
        # before each forecast, so a close that passes mid-run never
        # publishes. The framework reports the skip as a question error.
        close_time = question.close_time
        if close_time is None:
            return
        close = (
            close_time
            if close_time.tzinfo is not None
            else close_time.replace(tzinfo=timezone.utc)
        )
        remaining = (close - datetime.now(timezone.utc)).total_seconds()
        if remaining < MIN_MINUTES_TO_CLOSE * 60:
            raise ValueError(
                f"Skipping {question.page_url}: closes at "
                f"{close.isoformat()}, less than {MIN_MINUTES_TO_CLOSE} "
                "minutes left to publish."
            )

    async def _aggregate_predictions(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
        # Publish gate re-check: never publish past the close.
        self._ensure_time_to_publish(question)
        if isinstance(question, MultipleChoiceQuestion):
            # Framework default for MC is the MEAN; both winners publish
            # the MEDIAN (nostreambot's stacker tested no better than
            # median on 88 questions). Binary/numeric already aggregate
            # by median in the framework, so they go through super().
            return self._median_multiple_choice(predictions, question)
        return await super()._aggregate_predictions(predictions, question)

    @staticmethod
    def _median_multiple_choice(
        predictions: list[PredictedOptionList],
        question: MultipleChoiceQuestion,
    ) -> PredictedOptionList:
        option_names = [
            option.option_name for option in predictions[0].predicted_options
        ]
        medians: list[float] = []
        for name in option_names:
            probs = [
                option.probability
                for option_list in predictions
                for option in option_list.predicted_options
                if option.option_name == name
            ]
            medians.append(max(0.0, float(statistics.median(probs))))
        total = sum(medians)
        if total <= 0:
            medians = [1.0 / len(medians)] * len(medians)
        else:
            medians = [prob / total for prob in medians]
        return PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name=name, probability=prob)
                for name, prob in zip(option_names, medians)
            ]
        )

    @staticmethod
    def _safe_cdf_bounds(
        cdf: list[float], open_lower: bool, open_upper: bool
    ) -> list[float]:
        # Ported from Panshul42's numeric.py. Metaculus requires, for OPEN
        # bounds: cdf[0] >= 0.001 and cdf[-1] <= 0.999, and no single step
        # may exceed 0.59.
        cdf = list(cdf)
        if not cdf:
            return cdf
        if open_lower:
            cdf[0] = max(cdf[0], 0.001)
        if open_upper:
            cdf[-1] = min(cdf[-1], 0.999)
        for idx in range(len(cdf) - 1):
            if cdf[idx + 1] - cdf[idx] > 0.59:
                excess = cdf[idx + 1] - cdf[idx] - 0.59
                span = len(cdf) - idx - 1
                if span == 1:
                    cdf[idx + 1] -= excess
                else:
                    for k in range(span):
                        cdf[idx + 1 + k] -= excess * (1 - k / (span - 1))
                for j in range(idx + 1, len(cdf)):
                    if cdf[j] < cdf[j - 1]:
                        cdf[j] = cdf[j - 1]
        return cdf

    @classmethod
    def _enforce_cdf_legality(
        cls,
        percentile_list: list[Percentile],
        question: NumericQuestion | DateQuestion,
    ) -> list[Percentile]:
        # Dedupe percentile labels, enforce strictly increasing values,
        # then pin open-bound tails / cap steps at 0.59 via _safe_cdf_bounds.
        # With the 13 requested percentiles the repair is normally a no-op;
        # it guards degenerate parses (e.g. only 2 percentiles returned).
        by_label: dict[float, float] = {}
        for pct in percentile_list:
            by_label.setdefault(float(pct.percentile), float(pct.value))
        labels = sorted(by_label)
        if len(labels) < 2:
            return percentile_list
        values = [by_label[label] for label in labels]
        for i in range(1, len(values)):
            if values[i] <= values[i - 1]:
                values[i] = values[i - 1] + 1e-8
        heights = cls._safe_cdf_bounds(
            labels, question.open_lower_bound, question.open_upper_bound
        )
        for i in range(1, len(heights)):
            if heights[i] - heights[i - 1] < 5e-05:
                heights[i] = heights[i - 1] + 5e-05
        return [
            Percentile(percentile=height, value=value)
            for height, value in zip(heights, values)
        ]

    async def _fetch_market_snapshot(
        self, question: MetaculusQuestion
    ) -> str:
        # Prediction-market anchor (nostreambot market snapshot). Public
        # no-key APIs only (Polymarket Gamma + Kalshi). Any failure returns
        # "" so a dead venue never kills the run.
        def _fetch() -> str:
            keywords = _market_keywords(question.question_text)
            if not keywords:
                return ""
            rows: list[str] = []
            try:
                url = (
                    "https://gamma-api.polymarket.com/markets?limit=5&search="
                    + urllib.parse.quote(keywords)
                )
                with urllib.request.urlopen(url, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                for market in data if isinstance(data, list) else []:
                    try:
                        outcomes = market.get("outcomes") or []
                        prices = market.get("outcomePrices") or []
                        if outcomes and prices:
                            price = float(prices[0]) * 100
                            rows.append(
                                f'- Polymarket: "{market.get("question")}" — '
                                f"{outcomes[0]} {price:.0f}%"
                            )
                    except (TypeError, ValueError, IndexError):
                        continue
            except Exception as e:
                logger.warning(f"Polymarket snapshot failed: {e}")
            try:
                url = (
                    "https://api.elections.kalshi.com/trade-api/v2/markets"
                    "?limit=5&status=open&search=" + urllib.parse.quote(keywords)
                )
                with urllib.request.urlopen(url, timeout=8) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                for market in (data or {}).get("markets", []):
                    try:
                        price = market.get("last_price")
                        if price is None:
                            price = market.get("yes_bid")
                        if price is None:
                            continue
                        rows.append(
                            f'- Kalshi: "{market.get("title")}" — Yes '
                            f"{float(price):.0f}c"
                        )
                    except (TypeError, ValueError):
                        continue
            except Exception as e:
                logger.warning(f"Kalshi snapshot failed: {e}")
            if not rows:
                return ""
            today = datetime.now().strftime("%Y-%m-%d")
            header = (
                "## Prediction Market Snapshot\n"
                f"(live prices as of {today}; the snapshot is an anchor, not "
                "a forecast — see the market reading rules in the prompt)"
            )
            return header + "\n" + "\n".join(rows)

        return await asyncio.to_thread(_fetch)

    @staticmethod
    def _gap_fill_analyzer_prompt(
        question: MetaculusQuestion, first_pass_research: str
    ) -> str:
        # Ported shape from nostreambot's gap_fill_analyzer_prompt: a cheap
        # model lists the factual gaps whose resolution would move the
        # forecast, graded so code can drop a gap before its search is paid
        # for. Cap: GAP_FILL_MAX_GAPS, most forecast-moving first.
        options_line = ""
        if isinstance(question, MultipleChoiceQuestion):
            options_line = "Options (in resolution order): " + " | ".join(
                str(option) for option in question.options
            )
        return clean_indents(
            f"""
            You are a research-quality auditor. A forecaster has received first-pass research on a question. Your job: identify up to {GAP_FILL_MAX_GAPS} specific factual gaps where additional targeted search would meaningfully improve the forecast.

            Only flag a gap if resolving it would change how a superforecaster reasons about the question. DO NOT invent gaps for completeness: each gap is a paid search, and a slot spent on a gap that would not move the forecast is a slot not spent on one that would.

            Gap types to look for:
            1. Unread resolution sources — specific URLs, datasets, or reports named in resolution criteria or fine print that the first pass did not retrieve.
            2. Missing dates / chronology — first pass says "recently" or "this year" but the question turns on when exactly.
            3. Unaccessed flagged sources — first pass mentions a URL, PDF, or paywalled source it could not open.
            4. Missing quantitative specifics — first pass uses vague quantifiers ("high", "several", "many") where the question turns on a number.
            5. Unresolved contradictions — two sources disagree and the first pass did not fetch a tiebreaker.
            6. Missing base rate / reference class — anecdotes rather than historical frequency data.
            7. Missing expert opinion — a claim that should have a named expert or institution behind it but does not cite one.
            8. Stale first-pass info — first pass appears drawn from training data rather than current search.
            9. Missing counter-evidence — first pass is one-sided; a "consider the opposite" search would strengthen the forecast.

            ANSWERABLE NOW. Every gap must be answerable from sources that exist today. Never phrase a gap as a value on the resolution date ("what will <tracker> show on <date>"); rewrite it as the present-tense observable or drop it.

            NULL RESULTS ARE SEARCH OUTCOMES. Where the first pass says it searched and found nothing, treat that as an open question, not as an established negative fact.

            Order the gaps most forecast-moving first; the list ORDER is the ranking. GRADE EVERY GAP: answerable_now is false when the gap can only be answered by an observation not yet made; already_in_first_pass is true when the first-pass research already states the value or fact with its date; same_need_as is the position (1 = the first gap) of an earlier gap the same fact would answer, else null.

            Output STRICT JSON, nothing else, matching this schema exactly:
            {{"gaps": [{{"gap": "<specific factual question>", "why_matters": "<1 sentence>", "search_query": "<concise specific query>", "answerable_now": true/false, "already_in_first_pass": true/false, "same_need_as": <position or null>}}]}}
            If there are NO meaningful gaps, return {{"gaps": []}}.

            Question:
            {question.question_text}
            {options_line}

            Resolution criteria:
            {question.resolution_criteria or "(none provided)"}

            Fine print (often contains resolution sources):
            {question.fine_print or "(none provided)"}

            First-pass research:
            {first_pass_research}

            Return ONLY the JSON object. No preamble, no trailing commentary.
            """
        )

    async def _gap_fill_research(
        self, question: MetaculusQuestion, first_pass_research: str
    ) -> str:
        # One cheap LLM call lists the gaps; one SmartSearcher search per
        # surviving gap. Any failure returns "" (never kills the run).
        if not first_pass_research.strip():
            return ""
        try:
            analyzer = self.get_llm("summarizer", "llm")
            raw = await analyzer.invoke(
                self._gap_fill_analyzer_prompt(question, first_pass_research)
            )
        except Exception as e:
            logger.warning(f"Gap-fill analyzer failed: {e}")
            return ""
        try:
            start, end = raw.index("{"), raw.rindex("}") + 1
            gaps = json.loads(raw[start:end]).get("gaps", [])
        except (ValueError, AttributeError, TypeError) as e:
            logger.warning(f"Gap-fill JSON unparsable, skipping: {e}")
            return ""
        seen_queries: set[str] = set()
        kept: list[dict] = []
        for gap in gaps if isinstance(gaps, list) else []:
            if not isinstance(gap, dict):
                continue
            query = str(gap.get("search_query", "")).strip()
            if not query or query in seen_queries:
                continue
            if gap.get("answerable_now") is False:
                continue
            if gap.get("already_in_first_pass") is True:
                continue
            if gap.get("same_need_as") is not None:
                continue
            seen_queries.add(query)
            kept.append(gap)
            if len(kept) >= GAP_FILL_MAX_GAPS:
                break
        if not kept:
            return ""
        try:
            searcher_model = self.get_llm("summarizer", "string_name")
        except Exception as e:
            logger.warning(f"Gap-fill has no searcher model: {e}")
            return ""
        results: list[str] = []
        for gap in kept:
            try:
                searcher = SmartSearcher(
                    model=searcher_model,
                    temperature=0,
                    num_searches_to_run=1,
                    num_sites_per_search=5,
                    use_advanced_filters=False,
                )
                gap_prompt = clean_indents(
                    f"""
                    You are a research assistant resolving ONE specific factual gap for a forecaster.
                    Gap to resolve: {gap.get("gap")}
                    Suggested search query: {gap.get("search_query")}
                    This gap is from forecasting: {question.question_text}
                    Resolution criteria: {question.resolution_criteria or "(none provided)"}
                    Search the web for CURRENT, AUTHORITATIVE evidence addressing the gap. Report what you find, not what you think. Include inline citations. If the gap cannot be resolved with available sources, say so explicitly. DO NOT hallucinate sources. DO NOT produce a forecast.
                    """
                )
                found = await searcher.invoke(gap_prompt)
                if found and found.strip():
                    results.append(f"- Gap: {gap.get('gap')}\n  {found.strip()}")
            except Exception as e:
                logger.warning(
                    f"Gap-fill search failed for query "
                    f"'{gap.get('search_query')}': {e}"
                )
                continue
        if not results:
            return ""
        return (
            "## Targeted Gap-Fill (second pass)\n"
            + "\n".join(results)
        )

    ##################################### RESEARCH #####################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            self._ensure_time_to_publish(question)
            sections: list[str] = []
            # 1) Primary: AskNews (free for registered bot makers). A dead
            # primary never kills the run — the other sources still run.
            try:
                primary = await self._run_primary_research(question)
            except Exception as e:
                logger.warning(
                    f"Primary research failed for {question.page_url}: {e}"
                )
                primary = ""
            if primary:
                sections.append(primary)
            # 2) Second source + market snapshot in parallel; either may fail.
            second, snapshot = await asyncio.gather(
                self._run_second_source_research(question),
                self._fetch_market_snapshot(question),
                return_exceptions=True,
            )
            if isinstance(second, str) and second.strip():
                sections.append("## Second-source research\n" + second.strip())
            if isinstance(snapshot, str) and snapshot.strip():
                sections.append(snapshot.strip())
            first_pass = "\n\n".join(sections)
            # 3) Gap-fill pass: one cheap call lists missing facts, one
            # search per gap. Any failure returns "".
            try:
                gap_fill = await self._gap_fill_research(question, first_pass)
            except Exception as e:
                logger.warning(f"Gap-fill failed for {question.page_url}: {e}")
                gap_fill = ""
            if gap_fill:
                sections.append(gap_fill)
            research = "\n\n".join(sections)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")
            return research

    async def _run_primary_research(self, question: MetaculusQuestion) -> str:
        researcher = self.get_llm("researcher")
        prompt = self._get_research_prompt(question, researcher)
        if isinstance(researcher, GeneralLlm):
            return await researcher.invoke(prompt)
        if (
            researcher == "asknews/news-summaries"
            or researcher == "asknews/deep-research/low-depth"
            or researcher == "asknews/deep-research/medium-depth"
            or researcher == "asknews/deep-research/high-depth"
        ):
            return await AskNewsSearcher().call_preconfigured_version(
                researcher, prompt
            )
        if isinstance(researcher, str) and researcher.startswith("smart-searcher"):
            model_name = researcher.removeprefix("smart-searcher/")
            searcher = SmartSearcher(
                model=model_name,
                temperature=0,
                num_searches_to_run=2,
                num_sites_per_search=10,
                use_advanced_filters=False,
            )
            return await searcher.invoke(prompt)
        if not researcher or researcher == "None" or researcher == "no_research":
            return ""
        return await self.get_llm("researcher", "llm").invoke(prompt)

    async def _run_second_source_research(
        self, question: MetaculusQuestion
    ) -> str:
        # Second parallel source (both winners fan out beyond AskNews).
        # Cheapest available: a SmartSearcher web pass with the cheap
        # summarizer model. Needs EXA_API_KEY; without it this raises and
        # the caller drops the section.
        prompt = self._get_research_prompt(
            question, self.get_llm("summarizer", "string_name")
        )
        searcher = SmartSearcher(
            model=self.get_llm("summarizer", "string_name"),
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
            use_advanced_filters=False,
        )
        return await searcher.invoke(prompt)

    @staticmethod
    def _get_research_prompt(
        question: MetaculusQuestion, researcher: str | GeneralLlm
    ) -> str:
        options_line = ""
        if isinstance(question, MultipleChoiceQuestion):
            # Research must see the ballot: without the candidate names,
            # search never retrieves the eventual winner (nostreambot q44952).
            options_line = "Options (in resolution order): " + " | ".join(
                str(option) for option in question.options
            )
        if GeneralLlm.to_model_name(researcher) == "asknews/news-summaries":
            prompt = question.question_text
            if options_line:
                prompt += "\n" + options_line
            return prompt

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question.question_text}

            {options_line}

            This question's outcome will be determined by the specific criteria below:
            {question.resolution_criteria}

            {question.fine_print}

            {SOURCE_TIER_TAG_INSTRUCTION}

            Do NOT report Polymarket/Kalshi/Manifold/PredictIt prices from search results: a dedicated live snapshot of those venues is provided separately, and search-indexed copies of their prices are usually days stale. Do report market-implied or crowd odds from sources OTHER than those four venues (e.g. Metaculus, Good Judgment Open, CME FedWatch, bookmakers) — always name the market and the date you observed the price.
            """
        )
        return prompt

    ##################################### BINARY QUESTIONS #####################################

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        # TWO-STAGE (Panshul42, highest-impact change): stage 1 builds the
        # outside view — reference class + base rate from the historical
        # evidence in the research; stage 2 adjusts that base rate with the
        # current evidence. Each forecaster sees ONLY its own stage-1
        # output (Panshul42's cross-wired contexts look like a bug).
        forecaster = self._next_forecaster_llm()
        self._ensure_time_to_publish(question)
        today = datetime.now().strftime("%Y-%m-%d")
        stage1_prompt = clean_indents(
            f"""
            You are currently analyzing a forecasting question to generate an outside view prediction.

            The forecasting question is:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            Additional fine-print:
            {question.fine_print}

            Today is {today}.

            Research (focus on the HISTORICAL and base-rate evidence in it; ignore the latest-news items for now — those come in stage 2):
            {research}

            The information has been sourced from the internet, so exercise healthy skepticism at your discretion.

            Your analysis should have the following components, referring to the historical context:
            (a) Source analysis: Briefly summarize each information source, evaluate source quality and data. Opinions are commonplace in writing. For each source, discern factual information from opinions; strongly consider only opinions originating from identifiable experts or entities.
            (b) Reference class analysis: Identify a few possible reference classes and evaluate respective suitabilities to the forecasting question. If applicable, choose the most suitable one.
            (c) Timeframe analysis: State the prediction timeframe (e.g., how many days/months from now?) and examine historical patterns over similar periods.
            (d) Justification: Integrate the above factors with other points you found relevant to write a justification for your outside view prediction.

            Subsequently, calibrate your outside view prediction, considering:
            (a) You aim to predict the true probability of an event occurring, not a hedged or overconfident projection of your beliefs.
            (b) Is there a rough figure in the sources you can tether your prediction to?
            (c) Small differences in probabilities can be significant: 90% is a 9:1 odds and 99% is a 99:1 odds.

            Format your answer as below:

            Analysis:
            {{Insert your analysis here}}

            Outside view calibration:
            {{Insert your calibration of your outside view prediction here.}}

            Outside View Prediction:
            [Provide your outside view prediction here to a 1% significance. It is not necessary for the prediction to be a neat multiple of 5%.]
            """
        )
        stage1_reasoning = await forecaster.invoke(stage1_prompt)
        logger.info(
            f"Outside view for URL {question.page_url}: {stage1_reasoning}"
        )

        stage2_prompt = clean_indents(
            f"""
            You are currently analyzing a forecasting question to generate a final, inside view prediction.

            Your interview question is:
            {question.question_text}

            Question background:
            {question.background_info}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            {question.fine_print}

            Your outside view analysis (stage 1 — your own base rate):
            {stage1_reasoning}

            Current information / research:
            {research}

            Today is {today}.

            {MEASURED_FAILURE_RULES}

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A brief description of a scenario that results in a No outcome.
            (d) A brief description of a scenario that results in a Yes outcome.

            Then weight the evidence for/against resolution using this system:
            - Strong (large shifts): multiple independent, reliable and identifiable sources confirming the same direction; direct causal mechanisms; historical patterns with strong predictive power; structural/institutional factors that are difficult to change.
            - Moderate: a single reliable and identifiable source with clear methodology; indirect but logical causal links; similar historical patterns with some differences; current trends with demonstrated momentum.
            - Weak (small adjustments): anecdotal evidence; speculative connections; limited historical parallels; short-term or volatile indicators.

            Justify how the current factors shift your outside view base rate. For large shifts, justify how current factors are sufficiently impactful to render the reference class non-analogous to current circumstances.
            {self._get_conditional_disclaimer_if_necessary(question)}

            You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

            Verify your forecast with this checklist while reasoning (add a shortened version to your answer):
            1. Paraphrase the title and resolution criteria in <30 words, including relevant time windows.
            2. State your outside view base rate from stage 1 — check that your final prediction is genuinely rooted to it.
            3. Consistency check: "{{your prediction}} out of 100 times, {{resolution criteria}} happens." Does this make sense?
            4. The three-five most significant pieces of evidence, in <20 words each — verify each is factually valid.
            5. Blind-spot statement: the one scenario most likely to make your forecast look silly in hindsight.
            6. Status quo outcome: consider whether a slight nudge toward it is advantageous.

            The last thing you write is your final answer as: "Probability: ZZ%", 0-100
            """
        )
        stage2_reasoning = await forecaster.invoke(stage2_prompt)
        return await self._parse_binary_reasoning(question, stage2_reasoning)

    async def _parse_binary_reasoning(
        self,
        question: BinaryQuestion,
        reasoning: str,
    ) -> ReasonedPrediction[float]:
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a probability forecast for a binary question.
            {self._create_resolved_question_parsing_message()}
            """
        )
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning,
            BinaryPrediction,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}."
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    ##################################### MULTIPLE CHOICE QUESTIONS #####################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        # TWO-STAGE (Panshul42): outside view first, then inside view.
        # Each forecaster sees ONLY its own stage-1 output.
        forecaster = self._next_forecaster_llm()
        self._ensure_time_to_publish(question)
        today = datetime.now().strftime("%Y-%m-%d")
        stage1_prompt = clean_indents(
            f"""
            You are currently analyzing a forecasting question to generate an outside view prediction.

            The forecasting question is:
            {question.question_text}

            The options are: {question.options}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            Additional fine-print:
            {question.fine_print}

            Today is {today}.

            Research (focus on the HISTORICAL and base-rate evidence in it; ignore the latest-news items for now — those come in stage 2):
            {research}

            The information has been sourced from the internet, so exercise healthy skepticism at your discretion.

            Your analysis should have the following components, referring to the historical context:
            (a) Source analysis: Briefly summarize each information source, evaluate source quality and date. Discern factual information from opinions; strongly consider only opinions originating from identifiable experts or entities.
            (b) Reference class analysis: Identify a few possible reference classes and evaluate respective suitabilities to the forecasting question. If applicable, choose the most suitable one.
            (c) Timeframe analysis: State the prediction timeframe and examine historical patterns over similar periods.
            (d) Justification: Integrate the above factors to write a justification for your outside view prediction.

            Subsequently, calibrate your outside view prediction, considering:
            (a) You aim to predict the true probability of events occurring, not a hedged or overconfident projection of your beliefs.
            (b) Are there previously established distributions concerning the options that you can tether your prediction to?
            (c) Small differences in probabilities can be significant: 90% is a 9:1 odds and 99% is a 99:1 odds.
            (d) Historically, what is the rate of upsets/unexpected outcomes in the domain of this forecasting question? How should this affect your probability distribution?

            Format your answer as below:

            Analysis:
            {{Insert your analysis here, following the above components.}}

            Outside view calibration:
            {{Insert your calibration of your outside view prediction here.}}

            Outside View Prediction:
            Write your final probabilities (to a 1% significance) for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        stage1_reasoning = await forecaster.invoke(stage1_prompt)
        logger.info(
            f"Outside view for URL {question.page_url}: {stage1_reasoning}"
        )

        stage2_prompt = clean_indents(
            f"""
            You are currently analyzing a forecasting question to generate a final, inside view prediction.

            Your interview question is:
            {question.question_text}

            The options are: {question.options}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your outside view analysis (stage 1 — your own base rates):
            {stage1_reasoning}

            Current information / research:
            {research}

            Today is {today}.

            {MEASURED_FAILURE_RULES}

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of a scenario that results in an unexpected outcome.

            Then weight the evidence for/against resolution using this system:
            - Strong (large shifts): multiple independent, reliable sources confirming the same direction; direct causal mechanisms; historical patterns with strong predictive power; structural/institutional factors that are difficult to change.
            - Moderate: a single reliable source with clear methodology; indirect but logical causal links; similar historical patterns with some differences; current trends with demonstrated momentum.
            - Weak (small adjustments): anecdotal evidence; speculative connections; limited historical parallels; short-term or volatile indicators.

            Justify how the current factors shift your outside view base rates. For large shifts, justify how current factors are sufficiently impactful to render the reference class non-analogous to current circumstances.
            {self._get_conditional_disclaimer_if_necessary(question)}
            You write your rationale remembering that (1) good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time, and (2) good forecasters leave some moderate probability on most options to account for unexpected outcomes.

            Verify your forecast with this checklist while reasoning (add a shortened version to your answer):
            1. Paraphrase the possible options and resolution criteria in <30 words, including relevant time windows.
            2. State your outside view base rate distribution from stage 1 — check that your final prediction is genuinely rooted to it.
            3. Consistency check: "Most likely category/categories = ____; Least likely = ____; does this make sense and align with my justification?"
            4. The three-five most significant pieces of evidence, in <20 words each — verify each is factually valid.
            5. Blind-spot statement: the one scenario most likely to make your forecast look silly in hindsight.
            6. Technicalities: probabilities between 0 and 100, summing to 100.

            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        stage2_reasoning = await forecaster.invoke(stage2_prompt)
        return await self._parse_multiple_choice_reasoning(
            question, stage2_reasoning
        )

    async def _parse_multiple_choice_reasoning(
        self,
        question: MultipleChoiceQuestion,
        reasoning: str,
    ) -> ReasonedPrediction[PredictedOptionList]:
        parsing_instructions = clean_indents(
            f"""
            Make sure that all option names are one of the following:
            {question.options}

            The text you are parsing may prepend these options with some variation of "Option" which you should remove if not part of the option names I just gave you.
            Additionally, you may sometimes need to parse a 0% probability. Please do not skip options with 0% but rather make it an entry in your final list with 0% probability.
            {self._create_resolved_question_parsing_message()}
            """
        )
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            num_validation_samples=self._structure_output_validation_samples,
            additional_instructions=parsing_instructions,
        )

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}."
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    ##################################### NUMERIC QUESTIONS #####################################

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        # TWO-STAGE (Panshul42): outside view first, then inside view.
        # 13 percentiles (nostreambot) instead of 6; CDF legality enforced
        # on parse (open tails 0.001/0.999, max step 0.59, strictly
        # increasing — ported from Panshul42's numeric.py).
        forecaster = self._next_forecaster_llm()
        self._ensure_time_to_publish(question)
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        today = datetime.now().strftime("%Y-%m-%d")
        percentile_lines = "\n".join(
            f"            Percentile {pct}: XX"
            for pct in NUMERIC_PERCENTILES
        )
        stage1_prompt = clean_indents(
            f"""
            You are currently analyzing a numeric forecasting question to generate an outside view prediction.

            The forecasting question is:
            {question.question_text}

            This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
            {question.resolution_criteria}

            Additional fine-print:
            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Today is {today}.

            {lower_bound_message}
            {upper_bound_message}

            Research (focus on the HISTORICAL and base-rate evidence in it; ignore the latest-news items for now — those come in stage 2):
            {research}

            The information has been sourced from the internet, so exercise healthy skepticism at your discretion.

            Your analysis should have the following components, referring to the historical context:
            (a) Source analysis: Briefly summarize each information source, evaluate source quality and date. Discern factual information from opinions; strongly consider only opinions originating from identifiable experts or entities.
            (b) Reference class analysis: Identify a few possible reference classes and evaluate respective suitabilities to the forecasting question. If applicable, choose the most suitable one.
            (c) Timeframe analysis: State the prediction timeframe and examine historical patterns over similar periods.
            (d) Justification: Integrate the above factors to write a justification for your outside view prediction.

            Subsequently, calibrate your outside view prediction, considering:
            (a) You aim to predict a true probability distribution, not a hedged smooth distribution or an overconfident extremely narrow distribution. In your thinking, always consider ranges over singular values.
            (b) Are there previously established distributions that you can tether your prediction to?
            (c) Small changes in percentile location values can disproportionately reshape the slope and overall distribution of the extrapolated CDF, especially near the tails.
            (d) Historically, what is the rate of upsets/unexpected outcomes in the domain of this forecasting question? How should this affect your CDF distribution?

            It might be a good idea to set a wide 90/10 confidence interval to account for unknown unknowns.

            For your final outside view prediction, please keep in mind:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there.

            Format your answer as below:

            Analysis:
            {{Insert your analysis here, following the above components.}}

            Outside view calibration:
            {{Insert your calibration of your outside view prediction here.}}

            Outside View Prediction:
            Distribution:
{percentile_lines}
            """
        )
        stage1_reasoning = await forecaster.invoke(stage1_prompt)
        logger.info(
            f"Outside view for URL {question.page_url}: {stage1_reasoning}"
        )

        stage2_prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your outside view analysis (stage 1 — your own base-rate distribution):
            {stage1_reasoning}

            Current information / research:
            {research}

            Today is {today}.

            {lower_bound_message}
            {upper_bound_message}

            {MEASURED_FAILURE_RULES}

            Formatting Instructions:
            - Please notice the units requested and give your answer in these units (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there. The value for percentile 1 should always be less than the value for percentile 5, and so on.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            Then weight the evidence using this system:
            - Strong (large shifts): multiple independent, reliable sources confirming the same direction; direct causal mechanisms; historical patterns with strong predictive power; structural/institutional factors that are difficult to change.
            - Moderate: a single reliable source with clear methodology; indirect but logical causal links; similar historical patterns with some differences; current trends with demonstrated momentum.
            - Weak (small adjustments): anecdotal evidence; speculative connections; limited historical parallels; short-term or volatile indicators.

            Justify how the current factors shift your outside view base rates. For large distribution shifts, justify how current factors are sufficiently impactful to render the reference class non-analogous to current circumstances.
            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            Verify your forecast with this checklist while reasoning (add a shortened version to your answer):
            1. Paraphrase the target variable/resolution criteria in <20 words, including units and the time window.
            2. State your outside view base rate from stage 1 — check that your final prediction is genuinely rooted to it.
            3. Consistency check: "Implied median = ____; mean = ____; does this make sense and align with my justification?"
            4. The three-five most significant pieces of evidence, in <20 words each — verify each is factually valid.
            5. Blind-spot statement: the one scenario most likely to make your forecast look silly in hindsight, and whether it would push the outcome up or down.

            Essential formatting requirements:
            (a) For large numbers, please DO NOT output commas between numbers like 1,000,000. Instead, just write 1000000.
            (b) You MUST prefix the final percentiles with Distribution: as below.

            The last thing you write is your final answer as:
            "
            Distribution:
{percentile_lines}
            "
            """
        )
        stage2_reasoning = await forecaster.invoke(stage2_prompt)
        return await self._parse_numeric_reasoning(question, stage2_reasoning)

    async def _parse_numeric_reasoning(
        self,
        question: NumericQuestion,
        reasoning: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a numeric question.
            - This text is trying to answer the numeric question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - When parsing the text, please make sure to give the values (the ones assigned to percentiles) in terms of the correct units.
            - The units for the forecast are: {question.unit_of_measure}
            - Your work will be shown publicly with these units stated verbatim after the numbers your parse.
            - As an example, someone else guessed that the answer will be between {question.lower_bound} {question.unit_of_measure} and {question.upper_bound} {question.unit_of_measure}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - If the answer doesn't give the answer in the correct units, you should parse it in the right units. For instance if the answer gives numbers as $500,000,000 and units are "B $" then you should parse the answer as 0.5 (since $500,000,000 is $0.5 billion).
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            - Turn any values that are in scientific notation into regular numbers.
            """
        )
        percentile_list: list[Percentile] = await structure_output(
            reasoning,
            list[Percentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )
        percentile_list = self._enforce_cdf_legality(percentile_list, question)
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    ##################################### DATE QUESTIONS #####################################

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        # Single-stage (no two-stage date prompts in the winner repos), but
        # with a rotating forecaster, the measured-failure rules, 13
        # percentiles, and CDF legality enforcement on parse.
        forecaster = self._next_forecaster_llm()
        self._ensure_time_to_publish(question)
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        percentile_lines = "\n".join(
            f"            Percentile {pct}: YYYY-MM-DD"
            for pct in NUMERIC_PERCENTILES
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster interviewing for a job.

            Your interview question is:
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            {MEASURED_FAILURE_RULES}

            Formatting Instructions:
            - This is a date question, and as such, the answer must be expressed in terms of dates.
            - The dates must be written in the format of YYYY-MM-DD. If hours matter, please append the date with the hour in UTC and military time: YYYY-MM-DDTHH:MM:SSZ. No other formatting is allowed.
            - Always start with a lower date chronologically and then increase from there.
            - Do NOT forget this. The dates must be written in chronological order starting at the earliest time at percentile 1 and increasing from there.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            {self._get_conditional_disclaimer_if_necessary(question)}
            You remind yourself that good forecasters are humble and set wide 90/10 confidence intervals to account for unknown unknowns.

            The last thing you write is your final answer as:
            "
{percentile_lines}
            "
            """
        )
        reasoning = await forecaster.invoke(prompt)
        return await self._parse_date_reasoning(question, reasoning)

    async def _parse_date_reasoning(
        self,
        question: DateQuestion,
        reasoning: str,
    ) -> ReasonedPrediction[NumericDistribution]:
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        parsing_instructions = clean_indents(
            f"""
            The text given to you is trying to give a forecast distribution for a date question.
            - This text is trying to answer the question: "{question.question_text}".
            {self._create_single_distribution_parsing_message(question)}
            {self._create_resolved_question_parsing_message()}
            - As an example, someone else guessed that the answer will be between {question.lower_bound} and {question.upper_bound}, so the numbers parsed from an answer like this would be verbatim "{question.lower_bound}" and "{question.upper_bound}".
            - The output is given as dates/times please format it into a valid datetime parsable string. Assume midnight UTC if no hour is given.
            - If percentiles are not explicitly given (e.g. only a single value is given) please don't return a parsed output, but rather indicate that the answer is not explicitly given in the text.
            """
        )
        date_percentile_list: list[DatePercentile] = await structure_output(
            reasoning,
            list[DatePercentile],
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
            num_validation_samples=self._structure_output_validation_samples,
        )

        percentile_list = [
            Percentile(
                percentile=percentile.percentile,
                value=percentile.value.timestamp(),
            )
            for percentile in date_percentile_list
        ]
        percentile_list = self._enforce_cdf_legality(percentile_list, question)
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}."
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_resolved_question_parsing_message(self) -> str:
        return "- If the text concludes that the question has already resolved *in the past* (i.e. it treats the question as decided rather than something to forecast), please DO NOT return a parsed output, even if a final forecast is also given. Instead indicate that the answer is not explicitly given in the text.\n"

    def _create_single_distribution_parsing_message(
        self, question: NumericQuestion | DateQuestion
    ) -> str:
        message = (
            "- The text may contain multiple percentile distributions (e.g. forecasts for several related questions/entities, or intermediate drafts before a final answer). You must return exactly ONE distribution: the single final distribution that answers the question stated above.\n"
            "- Never merge or concatenate percentile lists that refer to different entities, options, or scenarios. Each percentile should appear at most once in your output.\n"
            "- If there are multiple final distributions and you cannot tell which one answers the question stated above, do not guess or combine them. Instead indicate that the answer is not explicitly given in the text."
        )
        if question.group_question_option is not None:
            message += (
                f'\n- This question is specifically about "{question.group_question_option}" (one subquestion within a group of related questions). '
                f'If the text gives distributions for multiple subjects, return only the distribution for "{question.group_question_option}".'
            )
        return message

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion | DateQuestion
    ) -> tuple[str, str]:
        if isinstance(question, NumericQuestion):
            if question.nominal_upper_bound is not None:
                upper_bound_number = question.nominal_upper_bound
            else:
                upper_bound_number = question.upper_bound
            if question.nominal_lower_bound is not None:
                lower_bound_number = question.nominal_lower_bound
            else:
                lower_bound_number = question.lower_bound
            unit_of_measure = question.unit_of_measure
        elif isinstance(question, DateQuestion):
            upper_bound_number = question.upper_bound.date().isoformat()
            lower_bound_number = question.lower_bound.date().isoformat()
            unit_of_measure = ""
        else:
            raise ValueError()

        if question.open_upper_bound:
            upper_bound_message = f"The question creator thinks the number is likely not higher than {upper_bound_number} {unit_of_measure}."
        else:
            upper_bound_message = f"The outcome can not be higher than {upper_bound_number} {unit_of_measure}."

        if question.open_lower_bound:
            lower_bound_message = f"The question creator thinks the number is likely not lower than {lower_bound_number} {unit_of_measure}."
        else:
            lower_bound_message = f"The outcome can not be lower than {lower_bound_number} {unit_of_measure}."
        return upper_bound_message, lower_bound_message

    ##################################### CONDITIONAL QUESTIONS #####################################

    async def _run_forecast_on_conditional(
        self, question: ConditionalQuestion, research: str
    ) -> ReasonedPrediction[ConditionalPrediction]:
        parent_info, full_research = await self._get_question_prediction_info(
            question.parent, research, "parent"
        )
        child_info, full_research = await self._get_question_prediction_info(
            question.child, research, "child"
        )
        yes_info, full_research = await self._get_question_prediction_info(
            question.question_yes, full_research, "yes"
        )
        no_info, full_research = await self._get_question_prediction_info(
            question.question_no, full_research, "no"
        )
        full_reasoning = clean_indents(
            f"""
            ## Parent Question Reasoning
            {parent_info.reasoning}
            ## Child Question Reasoning
            {child_info.reasoning}
            ## Yes Question Reasoning
            {yes_info.reasoning}
            ## No Question Reasoning
            {no_info.reasoning}
        """
        )
        full_prediction = ConditionalPrediction(
            parent=parent_info.prediction_value,  # type: ignore
            child=child_info.prediction_value,  # type: ignore
            prediction_yes=yes_info.prediction_value,  # type: ignore
            prediction_no=no_info.prediction_value,  # type: ignore
        )
        return ReasonedPrediction(
            reasoning=full_reasoning, prediction_value=full_prediction
        )

    async def _get_question_prediction_info(
        self, question: MetaculusQuestion, research: str, question_type: str
    ) -> tuple[ReasonedPrediction[PredictionTypes | PredictionAffirmed], str]:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        previous_forecasts = question.previous_forecasts
        if (
            question_type in ["parent", "child"]
            and previous_forecasts
            and question_type not in self.force_reforecast_in_conditional
        ):
            # TODO: add option to not affirm current parent/child forecasts, create new forecast
            previous_forecast = previous_forecasts[-1]
            current_utc_time = datetime.now(timezone.utc)
            if (
                previous_forecast.timestamp_end is None
                or previous_forecast.timestamp_end > current_utc_time
            ):
                pretty_value = DataOrganizer.get_readable_prediction(previous_forecast)  # type: ignore
                prediction = ReasonedPrediction(
                    prediction_value=PredictionAffirmed(),
                    reasoning=f"Already existing forecast reaffirmed at {pretty_value}.",
                )
                return (prediction, research)  # type: ignore
        info = await self._make_prediction(question, research)
        full_research = self._add_reasoning_to_research(research, info, question_type)
        return info, full_research  # type: ignore

    def _add_reasoning_to_research(
        self,
        research: str,
        reasoning: ReasonedPrediction[PredictionTypes],
        question_type: str,
    ) -> str:
        from forecasting_tools.data_models.data_organizer import DataOrganizer

        question_type = question_type.title()
        return clean_indents(
            f"""
            {research}
            ---
            ## {question_type} Question Information
            You have previously forecasted the {question_type} Question to the value: {DataOrganizer.get_readable_prediction(reasoning.prediction_value)}
            This is relevant information for your current forecast, but it is NOT your current forecast, but previous forecasting information that is relevant to your current forecast.
            The reasoning for the {question_type} Question was as such:
            ```
            {reasoning.reasoning}
            ```
            This is absolutely essential: do NOT use this reasoning to re-forecast the {question_type} question.
            """
        )

    def _get_conditional_disclaimer_if_necessary(
        self, question: MetaculusQuestion
    ) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return clean_indents(
            """
            As you are given a conditional question with a parent and child, you are to only forecast the **CHILD** question, given the parent question's resolution.
            You never re-forecast the parent question under any circumstances, but you use probabilistic reasoning, strongly considering the parent question's resolution, to forecast the child question.
            """
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the template forecasting bot")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="What to forecast on (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode

    check_environment(strict=True)
    publish_to_metaculus = True
    print_startup_banner(run_mode, will_publish=publish_to_metaculus)

    # Configure the bot. Forecaster models come from the
    # FallTemplateBot2026._llm_config_defaults override in main.py (one per
    # vendor via OpenRouter); the `llms=` block below stays commented out.
    # predictions_per_research_report=3: one prediction per forecaster
    # (round-robin). Two-stage prompting makes that 6 forecaster calls per
    # question (3 x outside view + 3 x inside view) vs 5 one-shot calls
    # before — plus 1 cheap gap-fill analyzer call and up to 4 gap searches.
    template_bot = FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish_to_metaculus,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        # llms={
        #     "default": GeneralLlm(
        #         model="openrouter/openai/gpt-4o",
        #         temperature=0.3,
        #         timeout=40,
        #         allowed_tries=2,
        #     ),
        #     "summarizer": "openai/gpt-4o-mini",
        #     "researcher": "asknews/news-summaries",
        #     "parser": "openai/gpt-4o-mini",
        # },
    )

    # Per-mode tournament URL shown in the summary banner footer. These
    # piggyback on the forecasting_tools SDK constants and need updating
    # whenever those rotate seasons.
    TOURNAMENT_URLS = {
        "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
        "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
        "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
    }

    # Dispatch on mode. Each branch produces a list of ForecastReport (or
    # exceptions, since return_exceptions=True) which then flows into the
    # summary printers below.
    client = MetaculusClient()
    if run_mode == "tournament":
        seasonal_tournament_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
        minibench_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_MINIBENCH_ID, return_exceptions=True
            )
        )
        forecast_reports = seasonal_tournament_reports + minibench_reports
    elif run_mode == "metaculus_cup":
        # The Metaculus Cup may be uninitialized near the start of a season
        # (Jan/May/Sep). MetaculusClient.ACX_2025_TOURNAMENT = 32564 and
        # MetaculusClient.AI_2027_TOURNAMENT_ID = "ai-2027" are also valid
        # targets here.
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                client.CURRENT_METACULUS_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # The bot-testing-area tournament contains all question types and is
        # the recommended target for smoke-testing your bot.
        # https://www.metaculus.com/tournament/bot-testing-area/
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                "bot-testing-area", return_exceptions=True
            )
        )

    template_bot.log_report_summary(forecast_reports)
    print_run_summary_banner(
        forecast_reports,
        will_publish=publish_to_metaculus,
        tournament_url=TOURNAMENT_URLS.get(run_mode),
    )
