from data_designer_slop_guard.core import Hyperparameters, analyze_text


CLEAN_TEXT = (
    "The bridge collapsed at 3:47 a.m. on a Tuesday. "
    "River water pushed through the gap in under a minute. "
    "Two cars stopped short of the edge. "
    "A third did not. "
    "The county engineer had flagged corrosion in a report eighteen months earlier. "
    "No repairs were scheduled."
)

SLOPPY_TEXT = (
    "In today's fast-paced world, it's worth noting that this groundbreaking, "
    "innovative framework represents a pivotal paradigm shift. It's important to note "
    "that it seamlessly leverages cutting-edge technology to foster holistic solutions. "
    "Furthermore, this revolutionary approach delves into the intricacies of the landscape, "
    "showcasing a comprehensive tapestry of robust capabilities. Let's dive in and explore "
    "this remarkable journey that transcends traditional boundaries. Additionally, the "
    "multifaceted nature of this testament to innovation harnesses the pinnacle of modern "
    "technology. Moreover, it navigates the spectrum of possibilities, orchestrating a "
    "seamless symphony of interconnected components. The trajectory of this odyssey "
    "underscores the formidable realm of next-generation systems."
)

SHORT_TEXT = "Hello world."


class TestAnalyzeText:
    def test_clean_text_scores_high(self):
        result = analyze_text(CLEAN_TEXT)
        assert result["score"] >= 80
        assert result["band"] == "clean"
        assert result["word_count"] > 0
        assert isinstance(result["violations"], list)
        assert isinstance(result["advice"], list)

    def test_sloppy_text_scores_low(self):
        result = analyze_text(SLOPPY_TEXT)
        assert result["score"] < 40
        assert result["band"] in ("heavy", "saturated")
        assert len(result["violations"]) > 5
        assert len(result["advice"]) > 3
        assert result["counts"]["slop_words"] > 0
        assert result["counts"]["slop_phrases"] > 0

    def test_short_text_returns_clean(self):
        result = analyze_text(SHORT_TEXT)
        assert result["score"] == 100
        assert result["band"] == "clean"
        assert result["violations"] == []

    def test_empty_text_returns_clean(self):
        result = analyze_text("")
        assert result["score"] == 100
        assert result["band"] == "clean"

    def test_custom_hyperparameters(self):
        mild = (
            "The team released the software update on Monday morning. "
            "Users reported faster load times and fewer crashes. "
            "The database migration took about three hours to complete. "
            "Several edge cases required manual correction afterward. "
            "Monitoring showed a notable improvement in response latency across all regions."
        )
        strict = Hyperparameters(decay_lambda=0.2)
        lenient = Hyperparameters(decay_lambda=0.005)
        assert analyze_text(mild, hyperparameters=strict)["score"] <= analyze_text(mild, hyperparameters=lenient)["score"]

    def test_result_shape(self):
        result = analyze_text(SLOPPY_TEXT)
        expected_keys = {"score", "band", "word_count", "violations", "counts", "total_penalty", "weighted_sum", "density", "advice"}
        assert expected_keys == set(result.keys())

    def test_advice_is_deduplicated(self):
        result = analyze_text(SLOPPY_TEXT)
        assert len(result["advice"]) == len(set(result["advice"]))

    def test_violation_structure(self):
        result = analyze_text(SLOPPY_TEXT)
        for v in result["violations"]:
            assert "type" in v
            assert "rule" in v
            assert "match" in v
            assert "context" in v
            assert "penalty" in v
            assert v["type"] == "Violation"

    def test_detects_ai_disclosure(self):
        text = (
            "As an AI language model, I don't have personal opinions on this topic. "
            "However, based on my training data, I can provide some information about "
            "the historical context of this event and its implications for modern policy."
        )
        result = analyze_text(text)
        assert result["counts"]["ai_disclosure"] > 0

    def test_detects_weasel_phrases(self):
        text = (
            "Many believe that the current approach is flawed. Experts suggest that a "
            "fundamental rethinking is needed. Studies show that alternative methods "
            "produce better results in controlled environments. Research suggests "
            "the gap will only widen without intervention."
        )
        result = analyze_text(text)
        assert result["counts"]["weasel"] > 0
